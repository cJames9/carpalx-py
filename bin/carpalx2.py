#!/home/martink/bin/perl
# %% imports
import argparse
import configparser
import os
import re
import math

import bin.util.fileutils as fileutils
# import bin.util.log as log
import logging

import sys
import copy
from PIL import Image, ImageDraw, ImageFont, ImageColor
import pandas as pd
import hashlib

import json
from itertools import islice
from collections import defaultdict, Counter
from random import randint, choice
from timeit import default_timer
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)

# %% carpalx
class Carpalx:
    configdir = None
    debuglevel = None
    report_filter = None
    draw_filter = None
    stdout_filter = None
    report_period = None
    keyboard_output = None
    draw_period = None
    pngfile_keyboard_output = None
    def __init__(self, configfile=None):
        self.configfile = configfile
        self.config = Carpalx.load_configuration(self.configfile)
        # options.configfile = config.get('main', 'configfile')

        self.keyboard_input = fileutils.parse_conf_file(self.config.get('kb_definition', 'keyboard_input'))
        # self.effort_model = fileutils.parse_conf_file(self.config.get('model', 'effort_model'))
        # self.effort_k_param = fileutils.parse_conf_file(effort_model.get('main', 'k_param'))
        # self.effort_weight_param = fileutils.parse_conf_file(effort_model.get('main', 'weight_param'))
        # self.effort_path_cost = fileutils.parse_conf_file(effort_model.get('main', 'path_cost'))
        # self.effort_finger_distance = fileutils.parse_conf_file(effort_model.get('main', 'finger_distance'))

        self.colors = fileutils.parse_conf_file(self.config.get('kb_parameters', 'colors'))
        self.kb_modes = fileutils.parse_conf_file(self.config.get('kb_parameters', 'modes'))
        
        self.populate_configuration()
        self.__class__.configdir = self.config['main']['configdir']
        self.__class__.debuglevel = self.config.getint('options', 'debug')
        self.__class__.report_filter = self.config.get('kb_report', 'report_filter')
        self.__class__.draw_filter = self.config.get('kb_report', 'draw_filter')
        self.__class__.stdout_filter = self.config.get('kb_report', 'stdout_filter')
        self.__class__.report_period = self.config.getint('kb_report', 'report_period')
        self.__class__.keyboard_output = self.config.get('kb_definition', 'keyboard_output')
        self.__class__.draw_period = self.config.getint('kb_report', 'draw_period')
        self.__class__.pngfile_keyboard_output = self.config.get('kb_parameters', 'pngfile_keyboard_output')

        self.actions = self.config.get('main', 'action').split(',')

        for action in self.actions:
            # logging.info(f'found action {action}')
            match action:
                case 'loadtriads':
                    # read the document and extract triads
                    #
                    # triads are adjacent three-key combinations parsed from the
                    # document based on the setting of the mode=MODE value
                    # (see <mode MODE> block for filters for the MODE).
                    #
                    logging.info(f'loading triad from corpus file {self.config.get('corpus', 'corpus')}')
                    keytriads = self.read_document(self.config.get('corpus', 'corpus'))
                case 'loadkeyboard':
                    # create a keyboard and the associated effort matrix
                    logging.info(f'loading keyboard from {self.config.get('kb_definition', 'keyboard_input')}')
                    keyboard = Keyboard(self.config.get('kb_definition', 'keyboard_input'), self.config.get('model', 'effort_model'))
                case 'reporttriads':
                    logging.info('reporting triad frequency')
                    self.report_triads(keytriads, keyboard.keyboard)
                case 'reportwordeffort':
                    logging.info('reporting word efforts')
                    self.report_word_effort(keyboard.keyboard)
                case 'drawinputkeyboard':
                    logging.info('drawing input keyboard')
                    self.draw_keyboard(keyboard.keyboard, self.config.get('kb_parameters', 'pngfile_keyboard_input'), {'title': f'{self.config.get("kb_definition", "keyboard_input")} layout'})
                    keyboard.print_keyboard()
                case 'drawoutputkeyboard':
                    logging.info('drawing output keyboard')
                    self.draw_keyboard(keyboard.keyboard, self.config.get('kb_parameters', 'pngfile_keyboard_output'), {'title': f'Optimized layout, runid {self.config.get("main", "runid")}'})
                    keyboard.print_keyboard()
                case s if s.startswith('reporteffort'):
                    # calculate the canonical effort associated with the original
                    # keyboard layout - the layout will be altered to try to minimize this
                    report_option = s.removeprefix('reporteffort')
                    logging.info('calculating effort')
                    charlist = self.read_document(self.config.get('corpus', 'corpus'), get_charlist=True)
                    # self.config['options']['memorize'] = 'no'
                    keyboard.report_keyboard_effort(keytriads, report_option, charlist, memorize=False)
                    # self.config['options']['memorize'] = 'yes'
                case 'optimize':
                    # optimize the keyboard layout to decrease the effort
                    logging.info('optimizing keyboard')
                    timer_start = default_timer()
                    keyboard = self.optimize_keyboard(keyboard, keytriads)
                    timer_stop = default_timer()
                    keyboard.print_keyboard()
                    logging.info('Total time spent optimizing: {:.3f}'.format(timer_stop-timer_start))
                case ['exit', 'quit']:
                    break
                case other:
                    raise KeyError(f'cannot understand action {action}')
        # sys.exit(0)
    
    @staticmethod
    def load_configuration(file):
        filepath = file
        filename = os.path.split(filepath)[-1]
        dir = os.path.dirname(filepath)
        found = False
        if os.path.exists(filepath):
            found = True
        else:
            import platform
            dirs = ['../etc/', './etc/']
            if platform.system() != 'Windows':
                dirs.extend([f'{os.environ["HOME"]}/.carpalx/etc', f'{os.environ["HOME"]}/.carpalx'])
            for dir in dirs:
                filepath = os.path.join(dir, filename)
                if os.path.exists(filepath):
                    found = True
                    break
        if not found:
            raise FileNotFoundError(f'cannot find or read configuration file {file}')

        config = fileutils.parse_conf_file(filepath)
        config['main']['configfile'] = filename
        config['main']['configdir'] = os.path.abspath(dir)

        return config
    
    def populate_configuration(self):
        # for key in vars(self.options):
        #     if getattr(self.options, key) is not None:
        #         if not 'options' in self.config:
        self.config['options'] = {}
        self.config['options']['nocache'] = 'no'
        self.config['options']['debug'] = '0'
        #         self.config['options'][key] = str(getattr(self.options, key))
        self._parse_conf_data()

    def _parse_conf_data(self):
        '''
        Any configuration fields of the form __XXX__ are parsed and replaced
        with eval(XXX). The configuration can therefore depend on itself.
        
        flag = 10
        note = __2*$CONF{flag}__ # would become 2*10 = 20
        '''
        for section in self.config.sections():
            for key, value in self.config[section].items():
                pattern = re.compile(r'__([^_].+?)__')
                cmd = pattern.search(value)
                if cmd is not None:
                    cmd = eval(cmd.group(1))
                    self.config[section][key] = re.sub(pattern, cmd, value)
                else:
                    pass

    def read_document(self, file, get_charlist=False):
        '''Read a document from file and create a list of of character triads.
        
        Triads are overlapping (more on overlapping below) 3-character combinations. Each triad is
        stored along with the number of times it appears in the document. All triads are stored,
        including overlapping triads.

        triads['aab'] = frequency of aab
        triads['abc'] = frequency of abc

        For example, if the document line is

        I am a very lazy dog with big ears.

        Then the triads will be

        i am              iam
          am a            ama
           m a v          mav
             a ve         ave
               ver        ver
                ery       ery
                 ry l     ryl
                  y la    yla
        ...

        and so on. Notice that spaces in the document are disregarded during construction of the triads.

        Depending on the parse mode, the input document undergoes some transformation before triads are
        constructed. Each mode must be defined using a <mode_def> block. Three modes are defined and
        you can add more.

        You can control how the triads are read by the <triad> block

        <triad>
        maxnum = 1000 # limit number of triads
        overlap = yes # if set to yes, a triad potentially begins at each character (triads overlap by maximum of 2 characters)
                        # if set to no, triads abut
        </triad>

        =over

        =item mode = perl

        If the mode is set to "perl", then all comment lines are disregarded. Comments are identified
        by lines that begin with #.

        =item mode = english

        English mode removes all non-alphanumeric characters before constructing triads.

        =item mode = letter

        All non-letter characters are removed and remaining letters are switched to lower case.'''

        # prepend script's path before relative paths to the input text
        file = fileutils.resolve_path(Carpalx.configdir, file)
        if not os.path.exists(file):
            raise FileNotFoundError(f'cannot find input document {file}')

        # die "no permission to read input document [$file]" unless -r $file;

        keytriads = defaultdict(int)
        triads_count = 0

        # read from corpus caches, if available
        file_cache = file + '.cache'
        file_cache_chars = file + '.chars.cache'

        if not self.config.getboolean('options', 'nocache'):
            if os.path.exists(file_cache) and not get_charlist:
                self.print_debug(1, f'reading triads from cache {file_cache}')
                with open(file_cache, 'r') as f:
                    cache = json.load(f)
                return cache
            elif os.path.exists(file_cache_chars) and get_charlist:
                self.print_debug(1, f'reading character list from cache {file_cache_chars}')
                with open(file_cache_chars, 'r') as f:
                    cache = json.load(f)
                return cache

        with open(file, 'r', encoding='ansi') as f:
            input_file = f.read().splitlines()
        charlist = []

        for idx, line in enumerate(input_file):
            # remove spaces from text - assume that a space does not influence effort of typing
            line = ''.join(line.split())
            if not line.strip():
                continue
            self.print_debug(2, 'read_document', 'pre-processed', line)

            mode = f'mode_def {self.config.get("corpus", "mode")}'
            mode_cfg = self.kb_modes[mode]

            if mode_cfg['force_case'] == 'lc':
                line = line.lower()
            elif mode_cfg['force_case'] == 'uc':
                line = line.upper()

            if self.kb_modes.has_option(mode, 'reject_char_rx'):
                rx = re.compile(mode_cfg['reject_char_rx'])
                line = re.sub(rx, '', line)
            if self.kb_modes.has_option(mode, 'reject_line_rx'):
                rx = re.compile(mode_cfg['reject_line_rx'])
                line = re.sub(rx, '', line)
                if not line:
                    self.print_debug(2, 'read_document', 'skipping line', line)
                    continue
            if self.kb_modes.has_option(mode, 'accept_line_rx'):
                rx = re.compile(mode_cfg['accept_line_rx'])
                if not re.findall(rx, line):
                    self.print_debug(2, 'read_document', 'skipping line', line)
                    continue
            self.print_debug(2, 'read_document', 'processed line', line)

            if get_charlist:
                charlist.extend(line)

            line_triads = [line[idx:idx+3] for idx, item in enumerate(line)]
            line_triads = [triad for triad in line_triads if len(triad) == 3]
            iter_triads = iter(line_triads)
            for triad in iter_triads:
                if len(set(triad)) == 1 and not self.kb_modes.getboolean(mode, 'accept_repeats'):
                    # identical triplet - skip if requested
                    self.print_debug(3, f'skipping repeated triad {triad}')
                    line_triads.remove(triad)
                    continue
                keytriads[triad] += 1
                triads_count += 1
                self.print_debug(3, f'accepting {triad}')
                # suppressed, treating below when ordering by frequency
                # if config.has_option('corpus', 'triads_max_num') and triads_count >= config.get('corpus', 'triads_max_num'):
                #     print_debug(1, f'limiting triads to {config.get("corpus", "triads_max_num")}')
                #     break
                # TODO: triads_overlap might be in config.corpus or config.options
                # also, review this modus operandi
                if not self.config.getboolean('corpus', 'triads_overlap'):
                    next(islice(iter_triads, 2, 2), None)

        # TODO: find where charlist_max_length might be defined
        if get_charlist:
            if not self.config.has_option('options', 'charlist_max_length'):
                return charlist
            elif len(charlist) >= self.config.getint('options', 'charlist_max_length'):
                return charlist

        logging.info(f'found {triads_count} triads ({len(keytriads)} unique)')
        
        # remove low-frequency triads
        if self.config.has_option('corpus', 'triads_min_freq'):
            min_freq = self.config.getint('corpus', 'triads_min_freq')
            for triad, freq in keytriads.copy().items():
                if freq < min_freq:
                    self.print_debug(1, f'removing rare triad {triad} freq {freq}')
                    del keytriads[triad]

        # limit number of triads
        # TODO: triads_max_num might be in config.corpus or config.options
        if self.config.has_option('corpus', 'triads_max_num'):
            self.print_debug(1, f'limiting triads to {self.config.get("corpus", "triads_max_num")}')
            triads_by_freq = dict(sorted(keytriads.items(), key=lambda item: item[1], reverse=True))
            for triad_to_del in list(triads_by_freq.keys())[self.config.getint('corpus', 'triads_max_num'):]:
                self.print_debug(1, f'removing triad {triad_to_del} freq {keytriads[triad_to_del]}')
                del keytriads[triad_to_del]

        if not os.path.exists(file_cache):
            logging.info(f'writing triads to cache {file_cache}')
            with open(file_cache, 'w') as cache_file:
                cache_file.write(json.dumps(keytriads))

        if not os.path.exists(file_cache_chars):
            # process the character list to create a frequency table of
            # characters and character transitions
            logging.info(f'writing character list to cache {file_cache_chars}')
            with open(file_cache_chars, 'w') as cache_chars:
                cache_chars.write(json.dumps(charlist))

        return keytriads
    
    @staticmethod
    def print_debug(level, *messages):
        messages = [str(msg) for msg in messages]
        if Carpalx.debuglevel >= level:
            print(f'DEBUG: {" ".join(messages)}')
    
    def report_triads(self, keytriads, keyboard=None):
        n = sum(keytriads.values())
        nc = 0
        for triad, freq in keytriads.items():
            nc += freq
            effort = keyboard.calculate_triad_effort(triad, dict(keyboard.k_param['effort'])) if keyboard is not None else None
            print('triad', triad, freq, freq/n, nc/n, 'effort', effort)
    
    def report_word_effort(self, keyboard):
        word_file = fileutils.resolve_path(Carpalx.configdir, self.config.get('wordstats', 'words'))
        if not os.path.exists(word_file):
            raise FileNotFoundError(f'cannot find word list file {word_file}')
        with open(word_file, 'r') as f:
            words = f.read().splitlines()
        # TODO: only for fast debugging
        # if self.config.getint('options', 'debug') > 1 and len(words) > 100_000:
        #     words = words[:100_000]
        if self.config.has_option('wordstats', 'wordlength'):
            min_len, max_len = self.config.get('wordstats', 'wordlength').split('-')
            min_len, max_len = int(min_len), int(max_len)
            for word in tqdm(words.copy(), desc='parsing words'):
                if len(word) < min_len or len(word) > max_len:
                    words.remove(word)
        word_effort = keyboard.rank_words(words)
        self.summarize_rank_words(word_effort)

    def summarize_rank_words(self, word_effort):
        '''
        Produce a summary of the word effort statistics.
        '''
        topN = 25
        # top 10
        sorted_desc = sorted(word_effort.items(), key=lambda item: item[1], reverse=True)
        sorted_asc = sorted(word_effort.items(), key=lambda item: item[1], reverse=False)
        if self.config.has_option('options', 'detail'):
            for word, effort in word_effort.items():
                print('word_effort', word, effort)
        top_hardest = list(islice(sorted_desc, topN))
        top_easiest = list(islice(sorted_asc, topN))
        print('wordreport', f'top {topN} hardest', ', '.join({f'{word}: {effort:.1f}' for word, effort in sorted_desc}))
        print('wordreport', f'top {topN} easiest', ', '.join({f'{word}: {effort:.1f}' for word, effort in sorted_asc}))
        # percentiles
        groups = 10
        for word, effort in top_hardest:
            print('wordreport group', 0, word, effort)
        for idx in range(groups):
            elemidx = int(idx * len(sorted_desc) / groups)
            words = sorted_desc[:elemidx + 10]
            cost = word_effort[words[-1][0]]
            print(f'wordreport percentile {int(100*idx/groups)} cost {cost:.1f}')
            for word in words:
                print('wordreport group', idx+1, word[0], word[1])
        for word in top_easiest:
            print('wordreport group', groups+1, word[0], word[1])

    def optimize_keyboard(self, keyboard, keytriads):
        '''
        Simulated annealing is used to search for a better keyboard layout.
        
        The function uses the list of triads, generated from the input text
        document, and an initial keyboard layout.
        '''
        
        annealing_options = self.config['annealing']
        mask_config = fileutils.parse_conf_file(self.config.get('kb_parameters', 'mask'))
        effort = keyboard.calculate_effort(keytriads)

        if 'iterations' in annealing_options:
            iterations = annealing_options.getint('iterations')
        else:
            iterations = 1000
        t0 = annealing_options.getint('t0')
        p0 = annealing_options.getint('p0')
        k = annealing_options.getint('k')
        # load up the mask - eligible keys for relocation
        mask = keyboard._parse_mask(mask_config)
        if not mask:
            raise Exception('cannot create mask')
        # create a list of all keys that can be relocated
        reloc_list = keyboard.make_relocatable_list(mask)
        
        if 'maxswap' in annealing_options:
            swap_range = (annealing_options.getint('minswaps'), annealing_options.getint('maxswaps'))
        else:
            swap_range = (annealing_options.getint('minswaps'), annealing_options.getint('minswaps'))
        action = annealing_options.get('action')

        update_count = 0
        last_reported_effort = 0
        original_keyboard = copy.deepcopy(keyboard)
        seen_digests = []
        effort = original_keyboard.calculate_effort(keytriads)
        
        for iteration in range(iterations):
            timer_start = default_timer()
            new_keyboard = {}
            
            if len(set(swap_range)) == 1:
                swap_num = swap_range[0]
            else:
                swap_num = randint(swap_range[0], swap_range[1]+1)
            if annealing_options.getboolean('onestep'):
                new_keyboard = original_keyboard._swap_keys(reloc_list, swap_num)
            else:
                new_keyboard = keyboard._swap_keys(reloc_list, swap_num)
            digest = new_keyboard.keyboard_digest()
            if digest in seen_digests:
                # already seen this layout - fetch next layout
                logging.info(f'skipping iteration {iteration}')
                continue
            seen_digests.append(digest)
            # this is breaking???
            new_effort = new_keyboard.calculate_effort(keytriads)
            deffort = new_effort - effort
            report = {}
            report['effort'] = effort
            report['neweffort'] = new_effort
            report['deffort'] = deffort
            t = t0 * math.exp(-iteration*k/iterations)
            p = p0 * math.exp(-abs(deffort)/t)
            p = 1 if p > 1 else p # float round-off
            report['t'] = t
            report['p'] = p
            keyboard_is_updated = 0
            
            if (action == 'minimize' and deffort < 0) \
                or (action == 'maximize' and deffort > 0):
                    # always accept layouts for which the effort is lower/higher (as prescribed by action)
                    effort = new_effort
                    keyboard = new_keyboard
                    report['move'] = 'better/accept'
                    keyboard_is_updated = 1
            elif randint(0, 5) < p:
                # sometimes accept layouts for which the effort is higher/lower (as prescribed by action)
                report['move'] = 'worse/accept'
                effort = new_effort
                keyboard = new_keyboard
                keyboard_is_updated = 1
            else:
                report['move'] = 'worse/reject'
            update_count += keyboard_is_updated
            
            make_report = False
            match Carpalx.report_filter:
                case 'all':
                    make_report = True
                case 'update':
                    make_report = True if report['move'].endswith('accept') else False
                case 'lower':
                    make_report = True if deffort < 0 else False
                case 'higher':
                    make_report = True if deffort > 0 else False
                case 'lower_monotonic':
                    make_report = True if not last_reported_effort or new_effort < last_reported_effort else False
                case 'higher_monotonic':
                    make_report = True if not last_reported_effort or new_effort > last_reported_effort else False
            
            make_draw = False
            match Carpalx.draw_filter:
                case 'all':
                    make_draw = True
                case 'update':
                    make_draw = True if report['move'].endswith('accept') else False
                case 'lower':
                    make_draw = True if deffort < 0 else False
                case 'higher':
                    make_draw = True if deffort > 0 else False
                case 'lower_monotonic':
                    make_draw = True if not last_reported_effort or new_effort < last_reported_effort else False
                case 'higher_monotonic':
                    make_draw = True if not last_reported_effort or new_effort > last_reported_effort else False
            
            if make_report:
                report['move'] += '/report'
            if make_draw:
                report['move'] += '/draw'
            
            stdout_report = False
            match Carpalx.stdout_filter:
                case 'all':
                    stdout_report = True
                case 'update':
                    stdout_report = True if report['move'].endswith('accept') else False
                case 'lower':
                    stdout_report = True if deffort < 0 else False
                case 'higher':
                    stdout_report = True if deffort > 0 else False
                case 'lower_monotonic':
                    stdout_report = True if not last_reported_effort or new_effort < last_reported_effort else False
                case 'higher_monotonic':
                    stdout_report = True if not last_reported_effort or new_effort > last_reported_effort else False
            
            parameters = {'t': t,
                          'iter': iteration,
                          'update_count': update_count,
                          'effort': effort,
                          'new_effort': new_effort,
                          'deffort': deffort}

            timer_stop = default_timer()
            time_elapsed = timer_stop - timer_start
            
            if stdout_report:
                new_keyboard.print_keyboard()
                print(f'iter {iteration} effort {effort:8.6f} -> {new_effort:8.6f} d {deffort:10.8f} p {p:10.8f} t {t:10.8f} {report["move"]} cpu {time_elapsed}')
            
            if make_report and not update_count % Carpalx.report_period == 0 :
                self.report_keyboard(new_keyboard, Carpalx.keyboard_output, parameters)
                last_reported_effort = new_effort
            
            if make_draw and not update_count % Carpalx.draw_period == 0:
                self.draw_keyboard(new_keyboard.keyboard, Carpalx.pngfile_keyboard_output, parameters)
                last_reported_effort = new_effort
        
        return keyboard
    
    def report_keyboard(self, keyboard, file, parameters):
        file = fileutils.resolve_path(Carpalx.configdir, file)
        with open(file, 'a') as f:
            output_parameters = self.config.get('kb_definition', 'keyboard_output_show_parameters').split(',')
            if 'current' in output_parameters:
                f.write('<current_parameters>\n')
                for parameter in parameters:
                    f.write(f'{parameter} = {parameters[parameter]}\n')
                f.write('</current_parameters>\n\n')
            if 'annealing' in output_parameters:
                f.write('<annealing_parameters>\n')
                for parameter in self.config['annealing']:
                    f.write(f'{parameter} = {self.config.get("annealing", parameter)}\n')
                f.write('</annealing_parameters>\n\n')
            
            f.write('<keyboard>\n')
            keys = []
            fingers = []
            for row_idx, row in keyboard.keyboard['key'].items():
                f.write(f'<row {row_idx+1}>\n')
                for col_idx, col in row.items():
                    lc, uc = col['lc'], col['uc']
                    if ord(lc) >= 97 and ord(lc) <= 122 and lc.upper() == uc:
                        keys.append(lc)
                    else:
                        # if lc == '#':
                        #     lc = '\\' + lc
                        # if uc == '#':
                        #     uc = '\\' + uc
                        keys.append(f'{lc}{uc}')
                    fingers.append(str(col['finger']))
                f.write(f'keys = {" ".join(keys)}\n')
                f.write(f'fingers = {" ".join(fingers)}\n')
                f.write('</row>\n')
            f.write('</keyboard>\n\n')

    def draw_keyboard(self, keyboard, file, parameters):
        '''
        Create an image of the keyboard
        '''
        # TODO: review row positioning

        file = fileutils.resolve_path(Carpalx.configdir, file)
        
        imageparamset = self.config.getint('kb_parameters', 'imageparamset')
        image_params = self.config[f'imageparamsetdef {imageparamset}']
        imagedetaillevel = self.config.getint('kb_parameters', 'imagedetaillevel')
        image_detail = self.config[f'imagedetaildef {imagedetaillevel}']
        colors = dict(self.config[f'color {imageparamset}'])
        
        keysize = image_params.getfloat('keysize')
        charxshift = image_params.getfloat('xshift')
        ucyshift = image_params.getfloat('ucyshift')
        lcyshift = image_params.getfloat('lcyshift')
        keymargin = image_params.getfloat('keyspacing')
        keymargin = keymargin * keysize if keymargin < 1 else keymargin
        shadow = image_params.getfloat('shadowsize')
        
        width = int((2+len(keyboard['key'][1])) * (keysize+keymargin))
        height = int(len(keyboard['key']) * (keysize+keymargin) + image_params.getfloat('bottommargin'))
        
        image = Image.new(mode='RGBA', size=(width, height), color=(255, 255, 255, 0))
        draw = ImageDraw.Draw(image)
        
        draw.rectangle([0, 0, width, height], fill=colors['background'])
        if image_detail.getboolean('imageborder'):
            draw.rectangle([0, 0, width - 1, height - 1], outline=colors['imageborder'])
        
        # get list of all unique costs
        costs = {}
        for row in keyboard['key']:
            for key in keyboard['key'][row]:
                total_effort = keyboard['key'][row][key]['effort']['total']
                costs[total_effort] = costs.get(total_effort, 0) + 1
        
        # rank ordered list of costs
        costs_ranks = sorted(costs.keys(), reverse=True)
        
        costs_colors = {}
        color_palette = [colors['effort_color_f'], colors['effort_color_i']]
        # keycolor_i = colors['effort_color_i']
        # keycolor_f = colors['effort_color_f']
        min_cost, max_cost = costs_ranks[-1], costs_ranks[0]
        
        for i, cost in enumerate(costs_ranks):
            rankcolor = self.gradient_color(min_cost, max_cost, cost, color_palette)
            # rankcolor = [int(max(keycolor_i[j] - i / (len(costs_ranks) - 1) * (keycolor_i[j] - keycolor_f[j])) ) for j in range(3)]
            colorname = f"rankcolor{i}"
            colors[colorname] = rankcolor  # RGB
            costs_colors[cost] = colorname
        
        for row in keyboard['key'].keys():
            keyy = (row + 1) * keymargin + row * keysize
            for col_idx, key in keyboard['key'][row].items():
                key_x = (1 + col_idx) * keymargin + keysize * col_idx
                cost = key['effort']['total']
                hand = key['hand']
                
                # Determine colors based on imagedetail
                keycolor = costs_colors[cost] if image_detail.getboolean('effortcolor') else colors['key']
                
                if image_detail.getboolean('keyshadow'):
                    draw.rectangle([key_x + shadow, keyy + shadow, key_x + keysize + shadow, keyy + keysize + shadow], fill=colors['keyshadow'])
                if image_detail.getboolean('fillkey'):
                    draw.rectangle([key_x, keyy, key_x + keysize, keyy + keysize], fill=keycolor)
                if image_detail.getboolean('keyborder'):
                    draw.rectangle([key_x, keyy, key_x + keysize, keyy + keysize], outline=colors['keyborder'])

                # Render text
                char_lc = key['lc']
                char_uc = key['uc']
                label_x = key_x + charxshift
                labely = keyy + ucyshift
                font = self.config.get('kb_parameters', 'font')
                
                if image_detail.get('upcase') == 'yes' or (not char_uc.isupper() and image_detail.get('upcase') == 'some'):
                    self.render_text(draw, font, label_x, labely, char_uc, image_params.getint('fontsize'), 'black', image_detail.getboolean('capitalize'))

                if image_detail.getboolean('lowcase'):
                    self.render_text(draw, font, label_x, labely + lcyshift, char_lc, image_params.getint('fontsize'), 'black')

                if image_detail.getboolean('effort'):
                    self.render_text(draw, font, key_x + keysize - 16, keyy + keysize - 5, f"{cost:.1f}", image_params.getint('fontsize'), 'black')
                if image_detail.getboolean('hand'):
                    self.render_text(draw, font, key_x + keysize - 7, keyy + keysize - 15, "R" if hand else "L", image_params.getint('fontsize'), 'black')
                if image_detail.getboolean('finger'):
                    self.render_text(draw, font, key_x + keysize - 7, keyy + keysize - 25, key['finger'], image_params.getint('fontsize'), 'black')

        if parameters and image_detail.getboolean('parameters'):
            self.render_parameters(draw, parameters, self.config.get('kb_parameters', 'fontc'), width, height, 'black')

        self.print_debug(1, 'creating keyboard image', file)
        os.path.normpath(file)
        image.save(file)

    def render_text(self, draw, font_path, x, y, text, size, color, capitalize=False):
        font = ImageFont.truetype(font_path, size)
        if capitalize:
            text = text.upper()
        draw.text((x, y), text, font=font, fill=color)

    def render_parameters(self, draw, parameters, font_path, width, height, color):
        text = []
        for parameter, value in sorted(parameters.items()):
            format_str = "%.5f" if isinstance(value, float) else "%s"
            text.append(f"{parameter} = {format_str % value}")

        self.render_text(draw, font_path, 5, height - 10, " :: ".join(text), 10, color)

    def gradient_color(self, minval, maxval, val, color_palette):
        """ Computes intermediate RGB color of a value in the range of minval
            to maxval (inclusive) based on a color_palette representing the range.
        """
        max_index = len(color_palette)-1
        delta = maxval - minval
        if delta == 0:
            delta = 1
        v = float(val-minval) / delta * max_index
        i1, i2 = int(v), min(int(v)+1, max_index)
        (r1, g1, b1) = ImageColor.getrgb(color_palette[i1])
        (r2, g2, b2) = ImageColor.getrgb(color_palette[i2])
        f = v - i1
        return int(r1 + f*(r2-r1)), int(g1 + f*(g2-g1)), int(b1 + f*(b2-b1))
    
    @staticmethod
    def histogram(table, title=None, datatitle=None, sort='index'):

        df = pd.Series(table).rename('Data').to_frame()
        df.index.rename(datatitle, inplace=True)
        df['Frequency'] = df['Data'] / df['Data'].sum() * 100
        
        if sort == 'values':
            df = df.sort_values('Data', ascending=False)
        elif sort == 'index':
            df = df.sort_index()
        
        df['Cumulative'] = df['Frequency'].cumsum()
        df = df.round(decimals=3)

        if title is not None:
            print(title)
            print("-" * 60)
        print(df)
        print('\n')

# %% keyboard
class Keyboard:
    def __init__(self, kb_definition=None, effort_model=None):
        self.kb_definition_file = kb_definition
        self.effort_model_file = effort_model
        
        self.effort_model = fileutils.parse_conf_file(self.effort_model_file)
        self.k_param = fileutils.parse_conf_file(self.effort_model.get('main', 'k_param'))
        self.weight_param = fileutils.parse_conf_file(self.effort_model.get('main', 'weight_param'))
        self.path_cost = fileutils.parse_conf_file(self.effort_model.get('main', 'path_cost'))
        self.finger_distance = fileutils.parse_conf_file(self.effort_model.get('main', 'finger_distance'))

        self.keyboard = self.create_keyboard(self.kb_definition_file)
        self._effortlookup = {}
    
    def create_keyboard(self, keyboard_type):
        '''
        Parses the keyboard layout and creates an array that keeps track of the keys, their positions,
        character assignments and typing effort.

        The keyboard array is indexed by row and column of the key and contains a hash

        my $keyboard = create_keyboard();

        $keyboard->{key}[row][col]
                                {lc}
                                {uc}
                                {row}
                                {col}
                                {effort}

        The keyboard layout is read from the <keyboard><row> blocks. The effort in the {key} part of
        the keyboard object is the canonical effort for the row,col combination as defined in
        <effort_row> plus any baseline and hand penalties.

        The keymap hash is a direct mapping between a character and its position and hand assignment
        on the keyboard

        $keyboard->{map}{CHAR}
                            {row}
                            {col}
                            {hand}
                            {effort}

        The effort in the {map} part of the keyboard object is the effort for the character, based
        on its row,col combination and includes the shift penalty.

        For the standard qwerty layout, look at the keyboard you're using right now (true for >99%
        of typists). For Dvorak layout, see http://www.mwbrooks.com/dvorak.

        '''

        keyboard = self._parse_keyboard_layout(keyboard_type)
        cost = self._parse_finger_distance()

        # merge the cost into the keyboard hash

        base_penalty = self.weight_param.getfloat('main', 'default')
        hand_weight = self.weight_param.getfloat('weight', 'hand')
        row_weight = self.weight_param.getfloat('weight', 'row')
        finger_weight = self.weight_param.getfloat('weight', 'finger')

        keyidx = 0
        for row_idx, row in keyboard['key'].items():
            for col_idx, col in keyboard['key'][row_idx].items():
                # this is the canonical cost of typing a key defined in the <effort_row> blocks
                base_effort = cost[row_idx][col_idx]
                finger      = col['finger']
                hand        = col['hand']
                row         = row_idx

                keyboard['key'][row_idx][col_idx].update({'idx': keyidx})

                if base_effort is None:
                    raise ValueError(f'create_keyboard - there is a key defined at row,col {row_idx},{col_idx} but no associated effort in the effort file')
                if finger is None:
                    raise ValueError(f'create_keyboard - there is a key defined at row,col {row_idx},{col_idx} but no finger assignment')
                if hand is None:
                    raise ValueError(f'create_keyboard - there is a key defined at row,col {row_idx},{col_idx} but no hand assignment')

                hand_descr = 'left' if hand == 0 else 'right'
                hand_penalty = self.weight_param.getfloat('hand', hand_descr)
                finger_penalty = float(self.weight_param.get('finger', hand_descr).split()[finger if hand == 0 else finger-5])
                row_penalty = self.weight_param.getfloat('row', str(row))

                penalty_effort = base_penalty \
                                + hand_weight * hand_penalty \
                                + row_weight * row_penalty \
                                + finger_weight * finger_penalty

                total_effort = self.k_param.getfloat('effort', 'kb') * base_effort + self.k_param.getfloat('effort', 'kp') * penalty_effort

                for case in ['lc', 'uc']:
                    Carpalx.print_debug(1,
                                'create_keyboard', 'effortassign',
                                'key', col[case],
                                'at row,col', row_idx, col_idx,
                                'hand', hand,
                                'row', row,
                                'finger', finger,
                                'base_effort', base_effort,
                                'base_penalty', base_penalty,
                                'hand_penalty', hand_weight, hand_penalty,
                                'row_penalty', row_weight, row_penalty,
                                'finger_penalty', finger_weight, finger_penalty,
                                'total_penalty', penalty_effort,
                                'total_effort', total_effort)

                    keyboard['map'][col[case]].update({'effort': {'total': total_effort,
                                                                'base':  base_effort,
                                                                'penalty': penalty_effort},
                                                    'idx': keyidx})
                keyboard['key'][row_idx][col_idx].update({'effort': {'total': total_effort,
                                                                    'base': base_effort,
                                                                    'penalty': penalty_effort}})

                keyidx += 1

        return keyboard
    
    def _parse_finger_distance(self):
        '''Load effort of hitting each key.

        The values here represent the baseline effort. You can add
        a effort offset in create_keyboard(), effectively adding
        a constant to all effort values.

        Home row keys require the least effort and therefore have
        the lowest effort. Keys assigned to the pinky have a high
        effort, especially if hitting them also requires wrist
        rotation (e.g. z requires wrist rotation but [ does not).

        '''
        cost = {}
        # row = 1
        leaf = dict(self.finger_distance['effort'])
        # die "typing effort for keyboard rows not defined - you need <effort_row ROW> blocks" unless ref($leaf) eq "HASH";
        for row in leaf.keys():
            keycost = leaf[row].split()
            cost.update({int(row.split()[-1])-1: [float(cost) for cost in keycost]})

        return cost

    def _parse_keyboard_layout(self, keyboard_file):
        '''
        Parse the <keyboard><row> blocks to determine the location of keys on the keyboard.

        Keyboard structure is stored as a hashref. Each key is stored by row/col position

        $keyboard->{key}[ROW][COL]{lc}     = lower case at ROW,COL
        $keyboard->{key}[ROW][COL]{uc}     = upper case at ROW,COL
        $keyboard->{key}[ROW][COL]{finger} = finger for hitting key at ROW,COL

        $keyboard->{map}{CHAR}{row}    = ROW for key CHAR
        $keyboard->{map}{CHAR}{col}    = COL for key CHAR
        $keyboard->{map}{CHAR}{case}   = CASE of key CHAR
        $keyboard->{map}{CHAR}{finger} = FINGER for hitting key CHAR

        '''

        keyboard_file = fileutils.resolve_path(Carpalx.configdir, keyboard_file)
        if not os.path.exists(keyboard_file):
            raise FileNotFoundError(f'cannot find keyboard definition file {keyboard_file}')

        keyboard = fileutils.parse_conf_file(keyboard_file)
        if not any([row for row in keyboard.sections() if row.startswith('row')]):
            raise KeyError(f'no keyboard row definitions in keyboard layout')

        keyboard_def = {'key': {}, 'map': {}}
        for section in keyboard.sections():
            if section.startswith('row'):
                row = int(section.split(' ')[-1]) -1
                keyboard_def['key'].update({row: {}})
                if 'keys' not in keyboard[section] or not keyboard.get(section, 'keys'):
                    raise KeyError(f'no keys defined in keyboard layout for {section}')
                if 'fingers' not in keyboard[section] or not keyboard.get(section, 'fingers'):
                    raise KeyError(f'no fingers defined in keyboard layout for {section}')

                keys = keyboard.get(section, 'keys').split()
                fingers = keyboard.get(section, 'fingers').split()
                fingers = [int(finger) for finger in fingers]

                keys = [[*key] for key in keys]
                for key in keys:
                    if ord(key[0]) >= 97 and ord(key[0]) <= 122:
                        key.insert(1, key[0].upper())
                    elif len(key) != 2:
                        raise KeyError(f'keyboard layout broken - non-letter key {key} must have two chasracters')

                for col, (key, finger) in enumerate(zip(keys, fingers)):
                    hand = 1 if finger > 5 else 0
                    Carpalx.print_debug(2, '_parse_keyboard_layout', 'keyassignment', 'lc', key[0], 'uc', key[1], 'to row,col', row, col)
                    keyboard_def['key'][row].update({col: {'lc': key[0], 'uc': key[1], 'finger': finger, 'hand': hand}})

                    for idx, case in enumerate(['lc', 'uc']):
                        keyboard_def['map'].update({key[idx]: {'row': row, 'col': col, 'case': case, 'finger': finger, 'hand': hand}})

        return keyboard_def
    
    def print_keyboard(self):
        '''
        Dump the keyboard layout to STDOUT
        '''
        # TODO: find where stdout_quiet might be defined
        # if config.has_option('option', 'stdout_quiet'):
        #     if config.getboolean('options', 'stdout_quiet'):
        #         return
        print('-'*60)
        for row in self.keyboard['key']:
            lcrow, ucrow = '', ''
            for col_idx, col in self.keyboard['key'][row].items():
                lcrow = lcrow + col['lc'] + ' '
                ucrow = ucrow + col['uc'] + ' '
            print(lcrow, ' '*(30-len(lcrow)), ucrow)
        print('-'*60)
    
    # def optimize_keyboard(self, keytriads, annealing_options, mask_config):
    #     '''
    #     Simulated annealing is used to search for a better keyboard layout.
        
    #     The function uses the list of triads, generated from the input text
    #     document, and an initial keyboard layout.
    #     '''
        
    #     effort = self.calculate_effort(keytriads)
    #     if 'iterations' in annealing_options:
    #         iterations = annealing_options.getint('iterations')
    #     else:
    #         iterations = 1000
    #     t0 = annealing_options.getint('t0')
    #     p0 = annealing_options.getint('p0')
    #     k = annealing_options.getint('k')
    #     # load up the mask - eligible keys for relocation
    #     mask = self._parse_mask(mask_config)
    #     if not mask:
    #         raise Exception('cannot create mask')
    #     # create a list of all keys that can be relocated
    #     reloc_list = self.make_relocatable_list(mask)
        
    #     if 'maxswap' in annealing_options:
    #         swap_range = (annealing_options.getint('minswaps'), annealing_options.getint('maxswaps'))
    #     else:
    #         swap_range = (annealing_options.getint('minswaps'), annealing_options.getint('minswaps'))
    #     action = annealing_options.get('action')

    #     update_count = 0
    #     last_reported_effort = 0
    #     original_keyboard = copy.deepcopy(self)
    #     seen_digests = []
    #     effort = original_keyboard.calculate_effort(keytriads)
        
    #     for iteration in range(iterations):
    #         timer_start = default_timer()
    #         new_keyboard = {}
            
    #         if len(set(swap_range)) == 1:
    #             swap_num = swap_range[0]    
    #         else:
    #             swap_num = randint(swap_range[0], swap_range[1]+1)
    #         if annealing_options.getboolean('onestep'):
    #             new_keyboard = original_keyboard._swap_keys(reloc_list, swap_num)
    #         else:
    #             new_keyboard = self._swap_keys(reloc_list, swap_num)
    #         digest = new_keyboard.keyboard_digest()
    #         if digest in seen_digests:
    #             # already seen this layout - fetch next layout
    #             continue
    #         seen_digests.append(digest)
    #         # this is breaking???
    #         new_effort = new_keyboard.calculate_effort(keytriads)
    #         deffort = new_effort - effort
    #         report = {}
    #         report['effort'] = effort
    #         report['neweffort'] = new_effort
    #         report['deffort'] = deffort
    #         t = t0 * math.exp(-iteration*k/iterations)
    #         p = p0 * math.exp(-abs(deffort)/t)
    #         p = 1 if p > 1 else p # float round-off
    #         report['t'] = t
    #         report['p'] = p
    #         keyboard_is_updated = 0
            
    #         if (action == 'minimize' and deffort < 0) \
    #             or (action == 'maximize' and deffort > 0):
    #                 # always accept layouts for which the effort is lower/higher (as prescribed by action)
    #                 effort = new_effort
    #                 self = new_keyboard
    #                 report['move'] = 'better/accept'
    #                 keyboard_is_updated = 1
    #         elif randint(0, 5) < p:
    #             # sometimes accept layouts for which the effort is higher/lower (as prescribed by action)
    #             report['move'] = 'worse/accept'
    #             effort = new_effort
    #             self = new_keyboard
    #             keyboard_is_updated = 1
    #         else:
    #             report['move'] = 'worse/reject'
    #         update_count += keyboard_is_updated
            
    #         make_report = False
    #         match Carpalx.report_filter:
    #             case 'all':
    #                 make_report = True
    #             case 'update':
    #                 make_report = True if report['move'].endswith('accept') else False
    #             case 'lower':
    #                 make_report = True if deffort < 0 else False
    #             case 'higher':
    #                 make_report = True if deffort > 0 else False
    #             case 'lower_monotonic':
    #                 make_report = True if not last_reported_effort or new_effort < last_reported_effort else False
    #             case 'higher_monotonic':
    #                 make_report = True if not last_reported_effort or new_effort > last_reported_effort else False
            
    #         make_draw = False
    #         match Carpalx.draw_filter:
    #             case 'all':
    #                 make_draw = True
    #             case 'update':
    #                 make_draw = True if report['move'].endswith('accept') else False
    #             case 'lower':
    #                 make_draw = True if deffort < 0 else False
    #             case 'higher':
    #                 make_draw = True if deffort > 0 else False
    #             case 'lower_monotonic':
    #                 make_draw = True if not last_reported_effort or new_effort < last_reported_effort else False
    #             case 'higher_monotonic':
    #                 make_draw = True if not last_reported_effort or new_effort > last_reported_effort else False
            
    #         if make_report:
    #             report['move'] += '/report'
    #         if make_draw:
    #             report['move'] += '/draw'
            
    #         stdout_report = False
    #         match Carpalx.stdout_filter:
    #             case 'all':
    #                 stdout_report = True
    #             case 'update':
    #                 stdout_report = True if report['move'].endswith('accept') else False
    #             case 'lower':
    #                 stdout_report = True if deffort < 0 else False
    #             case 'higher':
    #                 stdout_report = True if deffort > 0 else False
    #             case 'lower_monotonic':
    #                 stdout_report = True if not last_reported_effort or new_effort < last_reported_effort else False
    #             case 'higher_monotonic':
    #                 stdout_report = True if not last_reported_effort or new_effort > last_reported_effort else False
            
    #         parameters = {'t': t,
    #                       'iter': iteration,
    #                       'update_count': update_count,
    #                       'effort': effort,
    #                       'new_effort': new_effort,
    #                       'deffort': deffort}

    #         timer_stop = default_timer()
    #         time_elapsed = timer_stop - timer_start
            
    #         if stdout_report:
    #             new_keyboard.print_keyboard()
    #             print(f'iter {iteration} effort {effort:8.6f} -> {new_effort:8.6f} d {deffort:10.8f} p {p:10.8f} t {t:10.8f} {report["move"]} cpu {time_elapsed}')
            
    #         if make_report and not update_count % Carpalx.report_period == 0 :
    #             Carpalx.report_keyboard(Carpalx.self, self, Carpalx.keyboard_output, parameters)
    #             last_reported_effort = new_effort
            
    #         if make_draw and not update_count % Carpalx.draw_period == 0:
    #             Carpalx.draw_keyboard(self, Carpalx.pngfile_keyboard_output, parameters)
    #             last_reported_effort = new_effort
        
    #     return self
    
    def keyboard_digest(self):
        keys = []
        for row in self.keyboard['key']:
            for col in self.keyboard['key'][row]:
                for case in ['lc', 'uc']:
                    keys.append(''.join(self.keyboard['key'][row][col][case]))
        string = ':'.join(keys)
        string = string.encode('utf-8')
        hex_md5 = hashlib.new('md5')
        hex_md5.update(string)
        hex_md5 = hex_md5.hexdigest()
        
        return hex_md5
    
    def _swap_keys(self, reloc_list, n):
        '''
        Swap one or more pairs ($n randomly sampled pairs) of keys on the keyboard.
        
        Lower and upper case characters remain on the same key (e.g. no matter
        where 'a' is, A is always shift+a). This applies to both letter and
        non-letter characters (e.g. 1 and ! are always on the same key).

        This function returns a new keyboard object with the keys swapped.
        '''
        # TODO: implement __copy__ method to create copy from within class
        keyboard_copy = copy.deepcopy(self)
        if not n:
            n = 1
        for _ in range(n):
            # pick two random keyboard locations from the list of relocatable keys
            key1, key2 = 0, 0
            while key1 == key2:
                key1 = choice(reloc_list)
                key2 = choice(reloc_list)
            keyboard_copy._swap_key_pair(key1, key2)
        return keyboard_copy

    def _swap_key_pair(self, key1, key2):
        '''
        This function modifies $keyboard in place.
        '''
        # TODO: optimize!
        row1, col1 = key1
        row2, col2 = key2 
        
        k1lc = self.keyboard['key'][row1][col1]['lc']
        k1uc = self.keyboard['key'][row1][col1]['uc']
        k2lc = self.keyboard['key'][row2][col2]['lc']
        k2uc = self.keyboard['key'][row2][col2]['uc']
        
        self.keyboard['key'][row1][col1]['lc'] = k2lc
        self.keyboard['key'][row1][col1]['uc'] = k2uc
        self.keyboard['key'][row2][col2]['lc'] = k1lc
        self.keyboard['key'][row2][col2]['uc'] = k1uc
        
        m1lc = self.keyboard['map'][k1lc]
        m1uc = self.keyboard['map'][k1uc]
        m2lc = self.keyboard['map'][k2lc]
        m2uc = self.keyboard['map'][k2uc]
        
        self.keyboard['map'][k1lc] = m2lc
        self.keyboard['map'][k1uc] = m2uc
        self.keyboard['map'][k2lc] = m1lc
        self.keyboard['map'][k2uc] = m1uc
    
    def calculate_effort(self, keytriads, k_params=None, weight_params=None, memorize=True):
        '''
        Given a list of triads and the effort matrix, calculate the total carpal
        effort required to type the document from which the triads were generated.
        
        The effort is a non-negative number. The effort is a sum of the efforts
        for each triad. The total effort is normalized by the number of triads to
        remove dependency on document size.

        abcdefg
        abc     -> effort1
         bcd    -> effort2
          cde   -> effort3
            efg -> effort4
        -------    -------
        abcdefg -> total_effort = ( effort1 + effort2 + effort3 + effort4 ) /4

        Given a triad xyz, the effort is calculated by the following empirical expression

        effort = e = k1*effort(x) + k2*effort(x)*effort(y) + k3*effort(x)*effort(y)*effort(z) + k4*patheffort(x,y,z)
                = k1*effort(x)*[1 + effort(y)*(k2 + k3*effort(z))] + k4*patheffort(x,y,z)

        The form of this expression is motivated by the fact that the effort of
        three keystrokes is dependent on not only the individual identity of the
        keys but also alternation of hand, finger, row and column within the triad
        as well as presence of hard-to-type key combinations (e.g. zxc zqz awz ).
        For example, it is much easier to type "ttt" than "tbt", since the left
        forefinger must travel quite a distance in the latter example. Thus the
        insertion of the "b" character should impact the effort.

        In the first-order approximation k2=k3=k4=0 and the effort is simply the
        effort of typing the first key, effort(x). The individual effort of a key
        is defined in the <effort_row> blocks and is optionally modified by
        (a) shift penalty - CAPS are penalized and (b) hand penalty (e.g. you
        favour typing with your left hand). Since triads overlap, the first-order
        approximation for the entire document is the sum of the individual key
        efforts, without any long-range correlations.

        The addition of parameters k2 and k3 is designed to raise the effort of
        repeated difficult-to-type characters. This is where the notion of a triad
        comes into play. Notice that if effort(x) is zero, then the whole triad
        effort is zero.

        The patheffort(x,y,z) is a penalty which makes less desirable triads in
        which the keys do not follow a monotonic progression of columns, or triads
        which do not alternate hands. Once you try to type 'edc' on a qwerty
        keyboard, or 'erd' you will understand what I mean. The patheffort is a
        combination of two factors: hand alternation and column alternation. First,
        define a hand and column flag for a triad

        The definition of path effort here is arbitrary. I find that if the hands
        alternate between each keystroke, typing is easy (e.g. hf=0x). If both
        hands are used, but don't alternate then it's not as easy, particuarly
        when some of the columns in the triad are the same (e.g. same finger has
        to hit two keys like in "jeu"). If the same hand has to be used for three
        strokes then you're in trouble, particularly when some of the columns
        repeat. You can redefine the value of the path effort in <path_efforts>
        block.
        '''
        
        if not keytriads:
            raise ValueError('triads not defined')
        if not k_params:
            k_params = self.k_param
        if not weight_params:
            weight_params = self.weight_param
        
        total_effort = 0
        contributing_triads = 0
        k = dict(k_params['effort'])
        for triad in keytriads:
            triad_effort = self.calculate_triad_effort(triad, k, weight_params, memorize)
            num_triads = keytriads[triad]
            total_effort += triad_effort * num_triads
            contributing_triads += num_triads
            # if triad:
            #     print('calculate_effort', triad, num_triads, triad_effort, total_effort, contributing_triads)
        total_effort /= contributing_triads
        # print('calculate_effort_done', total_effort)
        # print('*'*80)
        return total_effort

    def calculate_triad_effort(self, triad, k_params, weight_params=None, memorize=True):
        if not weight_params:
            weight_params = self.weight_param
        
        k1, k2, k3 = (float(k_params['k1']), float(k_params['k2']), float(k_params['k3']))
        kb, kp, ks = (float(k_params['kb']), float(k_params['kp']), float(k_params['ks']))

        leaf = self.keyboard['map']

        # characters of the triad
        c1, c2, c3 = [*triad]
        i1, i2, i3 = (leaf[c1]['idx'], leaf[c2]['idx'], leaf[c3]['idx'])

        if memorize \
            and i1 in self._effortlookup \
            and i2 in self._effortlookup[i1] \
            and i3 in self._effortlookup[i1][i2]:
                return self._effortlookup[i1][i2][i3]

        # keyboard effort of each character
        be1, be2, be3    = (leaf[c1]['effort']['base'], leaf[c2]['effort']['base'], leaf[c3]['effort']['base'])
        pe1, pe2, pe3    = (leaf[c1]['effort']['penalty'], leaf[c2]['effort']['penalty'], leaf[c3]['effort']['penalty'])

        # finger of each character
        f1, f2, f3       = (leaf[c1]['finger'], leaf[c2]['finger'], leaf[c3]['finger'])
        # row of each character
        row1, row2, row3 = (leaf[c1]['row'], leaf[c2]['row'], leaf[c3]['row'])
        # hand of each character
        h1, h2, h3       = (leaf[c1]['hand'], leaf[c2]['hand'], leaf[c3]['hand'])
        # total triad effort is the sum of base effort product (finger distance) and penalty effort
        triad_effort     = kb * k1*be1 * ( 1 + k2*be2 * ( 1 + k3*be3 ) ) + kp * k1*pe1 * ( 1 + k2*pe2 * ( 1 + k3*pe3 ) )

        if ks:
            # hand, finger, row flags for stroke path
            # see http://mkweb.bcgsc.ca/carpalx/?typing_effort

            if h1 == h3 and h2 == h3: # same hand
                hand_flag = 2
            elif h1 == h3: # alternating hands
                hand_flag = 1
            else:
                hand_flag = 0

            if f1 > f2:
                if f2 > f3: # 1 > 2 > 3 - monotonic all different - pf=0
                    finger_flag = 0
                elif f2 == f3:
                    if c2 == c3: # 1 > 2 = 3 - monotonic some different - pf=1
                        finger_flag = 1
                    else:
                        finger_flag = 6
                elif f3 == f1:
                    finger_flag = 4
                elif f1 > f3 and f3 > f2: # rolling
                    finger_flag = 2
                else: # not monotonic all different - pf=3
                    finger_flag = 3
            elif f1 < f2:
                if f2 < f3: # 1 < 2 < 3 - monotonic all different - pf=0
                    finger_flag = 0
                elif f2 == f3:
                    if c2 == c3: # 1 < 2 = 3 - monotonic some different - pf=1
                        finger_flag = 1
                    else:
                        finger_flag = 6
                elif f3 == f1: # 1 = 3 < 2 - not monotonic some different - pf=2
                    finger_flag = 4
                elif f1 < f3 and f3 < f2: # rolling
                    finger_flag = 2
                else: # not monotonic all different - pf=3
                    finger_flag = 3
            elif f1 == f2:
                if f2 < f3 or f3 < f1: # 1 = 2 < 3 or 3 < 1 = 2 - monotonic some different - pf=1
                    if c1 == c2:
                        finger_flag = 1
                    else:
                        finger_flag = 6
                elif f2 == f3:
                    if c1 != c2 and c2 != c3 and c1 != c3:
                        finger_flag = 7
                    else: # 1 = 2 = 3 - all same - pf=4
                        finger_flag = 5

            dr = [(abs(row1 - row2), row1 - row2),
                (abs(row1 - row3), row1 - row3),
                (abs(row2 - row3), row2 - row3)]
            dr_sorted = sorted(dr, key=lambda x: (-x[0], x[1]))
            drmax_abs, drmax = dr_sorted[0]

            if row1 < row2:
                if row3 == row2: # 1 < 2 = 3 - downward with rep
                    row_flag = 1
                elif row2 < row3: # 1 < 2 < 3 - downward progression
                    row_flag = 4
                elif drmax_abs == 1:
                    row_flag = 3
                else: # all/some different - delta row > 1
                    if drmax < 0:
                        row_flag = 7
                    else:
                        row_flag = 5
            elif row1 > row2:
                if row3 == row2: # 1 > 2 = 3 - upward with rep
                    row_flag = 2
                elif row2 > row3: # 1 > 2 > 3 - upward
                    row_flag = 6
                elif drmax_abs == 1:
                    row_flag = 3
                else:
                    if drmax < 0:
                        row_flag = 7
                    else:
                        row_flag = 5
            else: # 1=2
                if row2 > row3: # 1 = 2 > 3 - upward with rep
                    row_flag = 2
                elif row2 < row3: # 1 = 2 < 3 - downward with rep
                    row_flag = 1
                else: # all same
                    row_flag = 0

            path_flag = f'{hand_flag}{row_flag}{finger_flag}'
            path_cost = weight_params.getfloat('main', 'path_offset') \
                        + float(self.path_cost.get('path', path_flag).split(' #')[0])
            triad_effort += ks * path_cost

            Carpalx.print_debug(1, f'triad {triad}',
                        f'keys {c1} {c2} {c3}',
                        f'base_effort {be1} {be2} {be3}',
                        f'penalty_effort {pe1} {pe2} {pe3}',
                        f'hand {h1} {h2} {h3}',
                        f'row {row1} {row2} {row3}',
                        f'finger {f1} {f2} {f3}',
                        f'ph {hand_flag} pr {row_flag} pf {finger_flag}',
                        f'path {path_flag} {path_cost} effort {triad_effort}')

        else:
            Carpalx.print_debug(1, f'triad {triad}',
                        f'keys {c1} {c2} {c3}',
                        f'base_effort {be1} {be2} {be3}',
                        f'penalty_effort {pe1} {pe2} {pe3}',
                        f'row {row1} {row2} {row3}',
                        f'hand {h1} {h2} {h3}',
                        f'ph - pr - pf - path - 0 effort {triad_effort}')

        if i1 not in self._effortlookup:
            self._effortlookup.update({i1: {i2: {i3: triad_effort}}})
        elif i2 not in self._effortlookup[i1]:
            self._effortlookup[i1].update({i2: {i3: triad_effort}})
        elif i3 not in self._effortlookup[i1][i2]:
            self._effortlookup[i1][i2].update({i3: triad_effort})
        else:
            self._effortlookup[i1][i2][i3] = triad_effort

        return triad_effort

    def report_keyboard_effort(self, keytriads, option, charlist, memorize=True):
        start_time = default_timer()
        effort = {}
        effort['all'] = self.calculate_effort(keytriads, memorize)

        # recall: triad effort is
        #
        # kb * k1*be1 * ( 1 + k2*be2 * ( 1 + k3*be3 ) ) +
        # kp * k1*pe1 * ( 1 + k2*pe2 * ( 1 + k3*pe3 ) ) +
        # ks * s
        #
        # be1,be2,be3 baseline efforts for first, second and third key in triad
        # pe1,pe2,pe3 penalty efforts for first, second and third key in triad
        # s stroke path

        local_k = copy.deepcopy(self.k_param)
        # baseline effort
        # kp=ks=0
        local_k['effort']['kp'] = '0'
        local_k['effort']['ks'] = '0'

        keyboard_new = Keyboard(self.kb_definition_file, self.effort_model_file)
        effort['base'] = keyboard_new.calculate_effort(keytriads, local_k)

        local_k = copy.deepcopy(self.k_param)
        # penalty effort
        # kb=ks=0
        local_k['effort']['kb'] = '0'
        local_k['effort']['ks'] = '0'
        keyboard_new = Keyboard(self.kb_definition_file, self.effort_model_file)
        effort['penalty'] = keyboard_new.calculate_effort(keytriads, local_k)

        local_k = copy.deepcopy(self.k_param)
        # stroke effort
        # kb=kp=0
        local_k['effort']['kb'] = '0'
        local_k['effort']['kp'] = '0'
        keyboard_new = Keyboard(self.kb_definition_file, self.effort_model_file)
        effort['path'] = keyboard_new.calculate_effort(keytriads, local_k)

        local_k = copy.deepcopy(self.k_param)
        local_weight = copy.deepcopy(self.weight_param)
        # hand penalty only
        local_k['effort']['kb'] = '0'
        local_k['effort']['ks'] = '0'
        local_weight['main']['default'] = '0'
        local_weight['weight']['row'] = '0'
        local_weight['weight']['finger'] = '0'
        keyboard_new = Keyboard(self.kb_definition_file, self.effort_model_file)
        effort['penalty_hand'] = keyboard_new.calculate_effort(keytriads, local_k, local_weight)

        local_k = copy.deepcopy(self.k_param)
        local_weight = copy.deepcopy(self.weight_param)
        # row penalty only
        local_k['effort']['kb'] = '0'
        local_k['effort']['ks'] = '0'
        local_weight['main']['default'] = '0'
        local_weight['weight']['hand'] = '0'
        local_weight['weight']['finger'] = '0'
        keyboard_new = Keyboard(self.kb_definition_file, self.effort_model_file)
        effort['penalty_row'] = keyboard_new.calculate_effort(keytriads, local_k, local_weight)

        local_k = copy.deepcopy(self.k_param)
        local_weight = copy.deepcopy(self.weight_param)
        # finger penalty only
        local_k['effort']['kb'] = '0'
        local_k['effort']['ks'] = '0'
        local_weight['main']['default'] = '0'
        local_weight['weight']['hand'] = '0'
        local_weight['weight']['row'] = '0'
        keyboard_new = Keyboard(self.kb_definition_file, self.effort_model_file)
        effort['penalty_finger'] = keyboard_new.calculate_effort(keytriads, local_k, local_weight)

        local_k = copy.deepcopy(self.k_param)
        local_weight = copy.deepcopy(self.weight_param)
        # one-key effort
        local_k['effort']['k2'] = '0'
        local_k['effort']['k3'] = '0'
        local_k['effort']['ks'] = '0'
        keyboard_new = Keyboard(self.kb_definition_file, self.effort_model_file)
        effort['k1'] = keyboard_new.calculate_effort(keytriads, local_k)

        local_k = copy.deepcopy(self.k_param)
        # two-key effort
        local_k['effort']['k3'] = '0'
        local_k['effort']['ks'] = '0'
        keyboard_new = Keyboard(self.kb_definition_file, self.effort_model_file)
        effort['k12'] = keyboard_new.calculate_effort(keytriads, local_k)

        local_k = copy.deepcopy(self.k_param)
        # three-key effort
        local_k['effort']['ks'] = '0'
        keyboard_new = Keyboard(self.kb_definition_file, self.effort_model_file)
        effort['k123'] = keyboard_new.calculate_effort(keytriads, local_k)

        efforts = {'k1': [effort['k1'],
                        100*sdiv(effort['k1'],effort['k123']),
                        100*sdiv(effort['k1'],effort['k123'])],
                'k12': [effort['k12'],
                        100*sdiv(effort['k12']-effort['k1'],effort['k123']),
                        100*sdiv(effort['k12'],effort['k123'])],
                'k123': [effort['k123'],
                            100*sdiv(effort['k123']-effort['k12'],effort['k123']),
                            100*sdiv(effort['k123'],effort['k123'])],
                'base': [effort['base'],
                            100*sdiv(effort['base'],effort['all']),
                            100*sdiv(effort['base'],effort['all'])],
                'penalty': [effort['penalty'],
                            100*sdiv(effort['penalty'],effort['all']),
                            100*sdiv(effort['base']+effort['penalty'],effort['penalty'])],
                'penalty_hand': [effort['penalty_hand'],
                                    100*sdiv(effort['penalty_hand'],effort['penalty']),
                                    100*sdiv(effort['penalty_hand'],effort['penalty'])],
                'penalty_row': [effort['penalty_row'],
                                100*sdiv(effort['penalty_row'],effort['penalty']),
                                100*sdiv(effort['penalty_hand']+effort['penalty_row'],effort['penalty'])],
                'penalty_finger': [effort['penalty_finger'],
                                    100*sdiv(effort['penalty_finger'],effort['penalty']),
                                    100*sdiv(effort['penalty_hand']+effort['penalty_row']+effort['penalty_finger'],effort['penalty'])],
                'path': [effort['path'],
                            100*sdiv(effort['path'],effort['all']),
                            100*sdiv(effort['base']+effort['penalty']+effort['path'],effort['all'])],
                'all': [effort['all'],
                        100*sdiv(effort['all'],effort['all']),
                        100*sdiv(effort['all'],effort['all'])]}

        print("Keyboard effort")
        print("-" * 60)

        kb_effort = pd.DataFrame(efforts).transpose().round(decimals=3)
        kb_effort.rename(columns={0: 'Absolute', 1: 'Relative', 2: 'Cumulative'}, inplace=True)
        print(kb_effort)
        
        if option == 'verybrief':
            return

        print('\n')

        stats = {}
        stats['row'] = defaultdict(int)
        stats['hand'] = defaultdict(int)
        stats['finger'] = defaultdict(int)
        for triad in keytriads:
            ntriad = keytriads[triad]
            char   = triad[0]
            row    = self.keyboard['map'][char]['row']
            hand   = self.keyboard['map'][char]['hand']
            finger = self.keyboard['map'][char]['finger']
            stats['row'][row] += ntriad
            stats['hand'][hand] += ntriad
            stats['finger'][finger] += ntriad

        Carpalx.histogram(stats['row'], 'keyboard row frequency', 'row')
        Carpalx.histogram(stats['hand'], 'keyboard hand frequency', 'hand')
        Carpalx.histogram(stats['finger'], 'keyboard finger frequency', 'finger')

        Carpalx.print_debug(1, "calculating runs")
        stats['charfreq'] = defaultdict(int)
        stats['charpairfreq'] = defaultdict(int)
        stats['run'] = defaultdict(dict)
        stats['run']['rowjump'] = defaultdict(int)
        
        charlist = [char for char in charlist if char in self.keyboard['map']]
        charpairs = ['.'.join(items) for items in list(zip(charlist, charlist[1:]))]
        stats['charfreq'] = Counter(charlist)
        stats['charpairfreq'] = Counter(charpairs)

        runlength = {}
        runlength['rowjump'] = 1
        runlength['finger'] = 1
        runlength['hand'] = 1
        runlength['row'] = 1

        for pair in charpairs:
            c1, c2 = pair.split('.')
            h1 = self.keyboard['map'][c1]['hand']
            h2 = self.keyboard['map'][c2]['hand']
            r1 = self.keyboard['map'][c1]['row']
            r2 = self.keyboard['map'][c2]['row']
            if h1 == h2 and r1 != r2:
                runlength['rowjump'] = runlength.get('rowjump', 0) + abs(r1-r2)
            else:
                stats['run']['rowjump'][runlength['rowjump']] += 1
                runlength['rowjump'] = stats['charpairfreq'][pair]
            
            for runtype in ['finger', 'hand', 'row']:
                cv = self.keyboard['map'][c2][runtype]
                cvp = self.keyboard['map'][c1][runtype]
                if cv == cvp:
                    runlength[runtype] += 1
                else:
                    # TODO: review if this won't be overwritten
                    if 'all' not in stats['run'][runtype]:
                        stats['run'][runtype]['all'] = defaultdict(dict)
                    stats['run'][runtype]['all'][runlength[runtype]] = stats['run'][runtype]['all'].get(runlength[runtype], 0) + stats['charpairfreq'][pair]
                    if runtype != 'finger':
                        if cvp not in stats['run'][runtype]:
                            stats['run'][runtype][cvp] = defaultdict(dict)
                        stats['run'][runtype][cvp][runlength[runtype]] = stats['run'][runtype][cvp].get(runlength[runtype], 0) + stats['charpairfreq'][pair]
                    runlength[runtype] = 1
        
        Carpalx.histogram(stats['run']['hand'][0], 'keyboard left hand run length', 'left_hand_run')
        Carpalx.histogram(stats['run']['hand'][1], 'keyboard right hand run length', 'right_hand_run')
        Carpalx.histogram(stats['run']['hand']['all'], 'keyboard hand run length', 'all_hand_run')
        Carpalx.histogram(stats['run']['row'][1], 'keyboard top row run length', 't_row_run')
        Carpalx.histogram(stats['run']['row'][2], 'keyboard home row run length', 'h_row_run')
        Carpalx.histogram(stats['run']['row'][3], 'keyboard bottom row run length', 'b_row_run')
        Carpalx.histogram(stats['run']['row']['all'], 'keyboard row run length', 'all_row_run')
        Carpalx.histogram(stats['run']['finger']['all'], 'keyboard finger run length', 'finger_run')
        Carpalx.histogram(stats['run']['rowjump'], 'keyboard same-hand row jump length', 'row_jump')
        Carpalx.histogram(stats['charfreq'], 'corpus character frequency', 'character_frequency', 'values')
        Carpalx.histogram(stats['charpairfreq'], 'corpus character pair frequency', 'character_pair_frequency', 'values')

        end_time = default_timer()
        elapsed_time = end_time - start_time
        print(f'Keyboard effort report generated in {elapsed_time:.2f} seconds')
    
        # def report_keyboard_effort(self, keytriads, option, charlist, memorize=True):
        # effort = {}
        # effort['all'] = self.calculate_effort(keytriads, memorize)

        # # recall: triad effort is
        # #
        # # kb * k1*be1 * ( 1 + k2*be2 * ( 1 + k3*be3 ) ) +
        # # kp * k1*pe1 * ( 1 + k2*pe2 * ( 1 + k3*pe3 ) ) +
        # # ks * s
        # #
        # # be1,be2,be3 baseline efforts for first, second and third key in triad
        # # pe1,pe2,pe3 penalty efforts for first, second and third key in triad
        # # s stroke path

        # local_k = copy.deepcopy(self.k_param)
        # # baseline effort
        # # kp=ks=0
        # local_k['effort']['kp'] = '0'
        # local_k['effort']['ks'] = '0'

        # keyboard_new = Keyboard(self.kb_definition_file, self.effort_model_file)
        # effort['base'] = keyboard_new.calculate_effort(keytriads, local_k)

        # local_k = copy.deepcopy(self.k_param)
        # # penalty effort
        # # kb=ks=0
        # local_k['effort']['kb'] = '0'
        # local_k['effort']['ks'] = '0'
        # keyboard_new = Keyboard(self.kb_definition_file, self.effort_model_file)
        # effort['penalty'] = keyboard_new.calculate_effort(keytriads, local_k)

        # local_k = copy.deepcopy(self.k_param)
        # # stroke effort
        # # kb=kp=0
        # local_k['effort']['kb'] = '0'
        # local_k['effort']['kp'] = '0'
        # keyboard_new = Keyboard(self.kb_definition_file, self.effort_model_file)
        # effort['path'] = keyboard_new.calculate_effort(keytriads, local_k)

        # local_k = copy.deepcopy(self.k_param)
        # local_weight = copy.deepcopy(self.weight_param)
        # # hand penalty only
        # local_k['effort']['kb'] = '0'
        # local_k['effort']['ks'] = '0'
        # local_weight['main']['default'] = '0'
        # local_weight['weight']['row'] = '0'
        # local_weight['weight']['finger'] = '0'
        # keyboard_new = Keyboard(self.kb_definition_file, self.effort_model_file)
        # effort['penalty_hand'] = keyboard_new.calculate_effort(keytriads, local_k, local_weight)

        # local_k = copy.deepcopy(self.k_param)
        # local_weight = copy.deepcopy(self.weight_param)
        # # row penalty only
        # local_k['effort']['kb'] = '0'
        # local_k['effort']['ks'] = '0'
        # local_weight['main']['default'] = '0'
        # local_weight['weight']['hand'] = '0'
        # local_weight['weight']['finger'] = '0'
        # keyboard_new = Keyboard(self.kb_definition_file, self.effort_model_file)
        # effort['penalty_row'] = keyboard_new.calculate_effort(keytriads, local_k, local_weight)

        # local_k = copy.deepcopy(self.k_param)
        # local_weight = copy.deepcopy(self.weight_param)
        # # finger penalty only
        # local_k['effort']['kb'] = '0'
        # local_k['effort']['ks'] = '0'
        # local_weight['main']['default'] = '0'
        # local_weight['weight']['hand'] = '0'
        # local_weight['weight']['row'] = '0'
        # keyboard_new = Keyboard(self.kb_definition_file, self.effort_model_file)
        # effort['penalty_finger'] = keyboard_new.calculate_effort(keytriads, local_k, local_weight)

        # local_k = copy.deepcopy(self.k_param)
        # local_weight = copy.deepcopy(self.weight_param)
        # # one-key effort
        # local_k['effort']['k2'] = '0'
        # local_k['effort']['k3'] = '0'
        # local_k['effort']['ks'] = '0'
        # keyboard_new = Keyboard(self.kb_definition_file, self.effort_model_file)
        # effort['k1'] = keyboard_new.calculate_effort(keytriads, local_k)

        # local_k = copy.deepcopy(self.k_param)
        # # two-key effort
        # local_k['effort']['k3'] = '0'
        # local_k['effort']['ks'] = '0'
        # keyboard_new = Keyboard(self.kb_definition_file, self.effort_model_file)
        # effort['k12'] = keyboard_new.calculate_effort(keytriads, local_k)

        # local_k = copy.deepcopy(self.k_param)
        # # three-key effort
        # local_k['effort']['ks'] = '0'
        # keyboard_new = Keyboard(self.kb_definition_file, self.effort_model_file)
        # effort['k123'] = keyboard_new.calculate_effort(keytriads, local_k)

        # efforts = {'k1': [effort['k1'],
        #                 100*sdiv(effort['k1'],effort['k123']),
        #                 100*sdiv(effort['k1'],effort['k123'])],
        #         'k12': [effort['k12'],
        #                 100*sdiv(effort['k12']-effort['k1'],effort['k123']),
        #                 100*sdiv(effort['k12'],effort['k123'])],
        #         'k123': [effort['k123'],
        #                     100*sdiv(effort['k123']-effort['k12'],effort['k123']),
        #                     100*sdiv(effort['k123'],effort['k123'])],
        #         'base': [effort['base'],
        #                     100*sdiv(effort['base'],effort['all']),
        #                     100*sdiv(effort['base'],effort['all'])],
        #         'penalty': [effort['penalty'],
        #                     100*sdiv(effort['penalty'],effort['all']),
        #                     100*sdiv(effort['base']+effort['penalty'],effort['penalty'])],
        #         'penalty_hand': [effort['penalty_hand'],
        #                             100*sdiv(effort['penalty_hand'],effort['penalty']),
        #                             100*sdiv(effort['penalty_hand'],effort['penalty'])],
        #         'penalty_row': [effort['penalty_row'],
        #                         100*sdiv(effort['penalty_row'],effort['penalty']),
        #                         100*sdiv(effort['penalty_hand']+effort['penalty_row'],effort['penalty'])],
        #         'penalty_finger': [effort['penalty_finger'],
        #                             100*sdiv(effort['penalty_finger'],effort['penalty']),
        #                             100*sdiv(effort['penalty_hand']+effort['penalty_row']+effort['penalty_finger'],effort['penalty'])],
        #         'path': [effort['path'],
        #                     100*sdiv(effort['path'],effort['all']),
        #                     100*sdiv(effort['base']+effort['penalty']+effort['path'],effort['all'])],
        #         'all': [effort['all'],
        #                 100*sdiv(effort['all'],effort['all']),
        #                 100*sdiv(effort['all'],effort['all'])]}

        # print("Keyboard effort")
        # print("-" * 60)

        # kb_effort = pd.DataFrame(efforts).transpose().round(decimals=3)
        # kb_effort.rename(columns={0: 'Absolute', 1: 'Relative', 2: 'Cumulative'}, inplace=True)
        # print(kb_effort)
        
        # if option == 'verybrief':
        #     return

        # print('\n')

        # stats = {}
        # stats['row'] = defaultdict(int)
        # stats['hand'] = defaultdict(int)
        # stats['finger'] = defaultdict(int)
        # for triad in keytriads:
        #     ntriad = keytriads[triad]
        #     char   = triad[0]
        #     row    = self.keyboard['map'][char]['row']
        #     hand   = self.keyboard['map'][char]['hand']
        #     finger = self.keyboard['map'][char]['finger']
        #     stats['row'][row] += ntriad
        #     stats['hand'][hand] += ntriad
        #     stats['finger'][finger] += ntriad

        # Carpalx.histogram(stats['row'], 'keyboard row frequency', 'row')
        # Carpalx.histogram(stats['hand'], 'keyboard hand frequency', 'hand')
        # Carpalx.histogram(stats['finger'], 'keyboard finger frequency', 'finger')

        # runlength = {}

        # Carpalx.print_debug(1, "calculating runs")
        # stats['charfreq'] = defaultdict(int)
        # stats['charpairfreq'] = defaultdict(int)
        # stats['run'] = defaultdict(dict)
        # stats['run']['rowjump'] = defaultdict(int)
        # for idx, char in enumerate(charlist):    
        #     if char not in self.keyboard['map']:
        #         charlist.remove(char)
        #         continue
        #     stats['charfreq'][char] += 1
        #     if idx == 0:
        #         runlength['rowjump'] = 1
        #         runlength['finger'] = 1
        #         runlength['hand'] = 1
        #         runlength['row'] = 1
        #     else:
        #         stats['charpairfreq'][f'{charlist[idx-1]}.{char}'] += 1
        #         try:
        #             h1 = self.keyboard['map'][char]['hand']
        #             h2 = self.keyboard['map'][charlist[idx-1]]['hand']
        #             r1 = self.keyboard['map'][char]['row']
        #             r2 = self.keyboard['map'][charlist[idx-1]]['row']
        #         except Exception as e:
        #             print(idx, char)
        #             print()
        #             raise e
        #         if h1 == h2 and r1 != r2:
        #             runlength['rowjump'] = runlength.get('rowjump', 0) + abs(r1-r2)
        #         else:
        #             stats['run']['rowjump'][runlength['rowjump']] += 1
        #             runlength['rowjump'] = 1
        #         if idx == len(charlist) - 1:
        #             stats['run']['rowjump'][runlength['rowjump']] = stats['run']['rowjump'].get(runlength['rowjump'], 0) + 1
        #         for runtype in ['finger', 'hand', 'row']:
        #             cv = self.keyboard['map'][char][runtype]
        #             cvp = self.keyboard['map'][charlist[idx-1]][runtype]
        #             if cv == cvp:
        #                 runlength[runtype] += 1
        #             else:
        #                 # TODO: review if this won't be overwritten
        #                 if 'all' not in stats['run'][runtype]:
        #                     stats['run'][runtype]['all'] = defaultdict(dict)
        #                 stats['run'][runtype]['all'][runlength[runtype]] = stats['run'][runtype]['all'].get(runlength[runtype], 0) + 1
        #                 if runtype != 'finger':
        #                     if cvp not in stats['run'][runtype]:
        #                         stats['run'][runtype][cvp] = defaultdict(dict)
        #                     stats['run'][runtype][cvp][runlength[runtype]] = stats['run'][runtype][cvp].get(runlength[runtype], 0) + 1
        #                 runlength[runtype] = 1
        #             if idx == len(charlist) - 1:
        #                 # TODO: review if this won't be overwritten
        #                 stats['run'][runtype]['all'][runlength[runtype]] = stats['run'][runtype]['all'].get(runlength[runtype], 0) + 1
        #                 if runtype != 'finger':
        #                     stats['run'][runtype][cvp][runlength[runtype]] = stats['run'][runtype][cvp].get(runlength[runtype], 0) + 1
        
        # Carpalx.histogram(stats['run']['hand'][0], 'keyboard left hand run length', 'left_hand_run')
        # Carpalx.histogram(stats['run']['hand'][1], 'keyboard right hand run length', 'right_hand_run')
        # Carpalx.histogram(stats['run']['hand']['all'], 'keyboard hand run length', 'all_hand_run')
        # Carpalx.histogram(stats['run']['row'][1], 'keyboard top row run length', 't_row_run')
        # Carpalx.histogram(stats['run']['row'][2], 'keyboard home row run length', 'h_row_run')
        # Carpalx.histogram(stats['run']['row'][3], 'keyboard bottom row run length', 'b_row_run')
        # Carpalx.histogram(stats['run']['row']['all'], 'keyboard row run length', 'all_row_run')
        # Carpalx.histogram(stats['run']['finger']['all'], 'keyboard finger run length', 'finger_run')
        # Carpalx.histogram(stats['run']['rowjump'], 'keyboard same-hand row jump length', 'row_jump')
        # Carpalx.histogram(stats['charfreq'], 'corpus character frequency', 'character_frequency', 'values')
        # Carpalx.histogram(stats['charpairfreq'], 'corpus character pair frequency', 'character_pair_frequency', 'values')
    
    def rank_words(self, words):
        '''
        given a list of words, return the words and associated
        efforts, as calculated using the triads of the words.
        '''
        word_effort = {}
        for word in tqdm(words, desc='ranking words'):
            triads = [word[idx:idx+3].lower() for idx, item in enumerate(word)]
            triads = [triad for triad in triads if len(triad) == 3]
            word_triads = Counter(triads)
            word_effort[word] = self.calculate_effort(word_triads)
            Carpalx.print_debug(1, 'word_effort', word, word_effort[word])
        return word_effort

    def _parse_mask(self, mask_config):
        '''
        Parse the keyboard mask, defined in <mask_row N> blocks in the
        configuration file. Each row block should contain an entry
        
        mask = M M M M M ...
        
        where M = 1|0 depending on whether you want the key to be
        eligible (1) or not eligible (0) for remapping. There should
        be as many Ms in each row as there are columns on the keyboard
        '''
        mask = {}
        rows = [section for section in mask_config.sections() if section.startswith('mask_row')]
        rows_idx = [int(row.split()[-1])-1 for row in rows]
        mask = dict.fromkeys(rows_idx)
        for row_idx, row in zip(rows_idx, rows):
            mask[row_idx] = defaultdict(int)
            row_mask = mask_config.get(row, 'mask').split()
            for col, col_mask in enumerate(row_mask):
                mask[row_idx][col] = int(col_mask)
        return mask

    def make_relocatable_list(self, mask):
        '''
        Based on the key mask generated by _parse_mask(), this function returns a
        list of all keys that can be relocated. The list is a set of row,col pairs.

        $list = [ ... [row,col], [row,col], ... ]
        '''
        reloc_list = []
        for row_idx, row in mask.items():
            for col, value in row.items():
                if value:
                    reloc_list.append([row_idx, col])
        return reloc_list

# %% header

# =pod

# =head1 NAME

# carpalx - given text input, determine optimal keyboard mapping to minimize typing effort based on a typing effort model

# =head1 SYNOPSIS

#   # all configuration read from etc/carpalx.conf
#   carpalx -keyboard_input keyboard.conf -keyboard_output keyboard-optimized.conf
#           -corpus corpus/words.txt
#           -action loadkeyboard,loadtriads,optimize
#           -conf etc/carpalx.conf
#           [-debug]

# =head1 DESCRIPTION

# carpalx is a keyboard layout optimizer. Given a training corpus
# (e.g. English text) and parameters that describe typing effort,
# carpalx uses simulated annealing to find a keyboard layout to minimize
# typing effort.

# Typing effort is modeled using three contributions. First, base effort
# is derived from finger travel distance. Second, row, hand and finger
# penalties are added to limit use of weaker fingers/hands and
# distinguish harder-to-reach keys. Third, stroke path effort is used to
# rate the effort based on finger, row and hand alternation (e.g. asd is
# much easier to type than sad).

# =head1 CONFIGURATION

# =head2 Configuration file name and path

# carpalx will look in the following locations for a configuration file

#   .
#   SCRIPT_BIN/../etc
#   SCRIPT_BIN/etc
#   SCRIPT_BIN/
#   ~/.carpalx/etc
#   ~/.carpalx

# where SCRIPT_BIN is the location of the carpalx script. If the name of
# the configuration file is not passed via -conf, then SCRIPT_NAME.conf
# is tried where SCRIPT_NAME is the name of the script. For example,

#   > cd carpalx-0.11
#   > bin/carpalx

# will attempt to find carpalx.conf in the above paths.

# Using -debug -debug will dump the configuration parameters.

#   > bin/carpalx -debug -debug

# =head2 Configuration structure

# The configuration file comprises variable-value pairs, which may be
# placed in blocks.

#   a = 1
#   <someblock>
#     b = 2
#     <anotherblock>
#     c = 3
#     </anotherblock>
#   </someblock>

# Combinations of related parameters (e.g. base effort, keyboard
# configuration) are stored in individual files
# (e.g. etc/mask/letters.conf) which are subsequently imported into
# the main configuration file using <<include>>

#   ...
#   <<include etc/mask/letters.conf>>
#   ...

# =head1 HISTORY

# =over

# =item * 0.10

# Packaged and versioned code.

# =item * 0.11

# Adjusted typing model to include weights for base, effort and stroke components.

# Improved clarity of effort reports.

# Improved consistency in configuration file.

# Added fonts/

# =item * 0.12

# Can now load a cache file basedon parsed corpus instead of original corpus.

# =back

# =head1 BUGS

# Report!

# =head1 AUTHOR

# Martin Krzywinski <martink@bcgsc.ca>
# http://mkweb.bcgsc.ca

# =head1 CONTACT

#   Martin Krzywinski
#   Genome Sciences Centre
#   100-570 W 7th Ave
#   Vancouver BC V5Z 4S6

# =cut

################################################################
#
# Copyright 2002-2014 Martin Krzywinski <martink@bcgsc.ca> http://mkweb.bcgsc.ca
#
# This file is part of the Genome Sciences Centre Perl code base.
#
# This script is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# This script is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this script; if not, write to the Free Software
# Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
#
################################################################

################################################################
#                             ___   __
#                            | \ \ / /
#   ___ __ _ _ __ _ __   __ _| |\ V /
#  / __/ _` | '__| '_ \ / _` | | > <
# | (_| (_| | |  | |_) | (_| | |/ . \
#  \___\__,_|_|  | .__/ \__,_|_/_/ \_\
#                | |
#                |_| v0.11
#
# carpalX - keyboard layout optimizer - save your carpals
#
# Face it, typing for 10 years can leave your hands looking like
# cranky twigs. Moving that pinky over and over again and rotating
# the wrist - ouch.
#
# CarpalX processes a document and computes the total carpal
# effort required to type it using a default qwerty keyboard
# layout. Effort of each key is defined by its location on the
# keyboard as well as the finger customarily responsible for hitting
# that key.
#
# Long-range effects (double and triple) key combinations
# contribute to 1st and 2nd order effort quantities. For example,
# it requires more effort to type aaazzzaaa than zaazaazaa, because
# the wrist rotation is prolonged and the pinky must extend to hit
# three z's in a row.
#
# You have the option to optimize the keyboard layout for the
# document. The simulated annealing method is used to determine
# a key layout that minimizes total effort.
#
################################################################

# %% sdiv

def sdiv(a, b):
    return a/b if b else 0

# %% find_action

# report the frequency and cumulative frequency of all triads

# sub find_action {
#   my ($rx,@actions) = @_;
#   if ( my ($action) = grep($rx,@actions) ) {
#     return $action;
#   } else {
#     return undef;
#   }
# }

# %% advance_actions

# sub advance_actions {
#   my ($action,@actions) = @_;
#   exit if $action =~ /exit|quit/;
#   my @newactions;
#   my $found;
#   for $a (@actions) {
#     if(! $found && $a eq $action) {
#       $found = 1;
#       next;
#     }
#     push @newactions, $a;
#   }
#   return @newactions;
# }

# %% validateconfiguration

################################################################

################################################################
#
# Housekeeeeeping, suppressed for now
#
################################################################

# sub validateconfiguration {
#   #printinfo($CONF{keyboard_output});
#   my $rootdir = dirname($FindBin::RealBin);
#   for my $type (qw(keyboard_output pngfile_keyboard_input pngfile_keyboard_output font fontc)) {
#     if($CONF{$type} !~ /^\//) {
#       $CONF{$type} = sprintf("%s/%s",$rootdir,$CONF{$type});
#     }
#     #printinfo($CONF{$type});
#   }
# }

# %% main

# def main():

#     parser = argparse.ArgumentParser()
#     opt = parser.add_argument_group('options')
#     opt.add_argument('-keyboard_input', action='store')
#     opt.add_argument('-keyboard_output', action='store')
#     opt.add_argument('-nocache', action='store_true')
#     opt.add_argument('-action', action='store')
#     opt.add_argument('-corpus', action='store')
#     opt.add_argument('-words', action='store')
#     opt.add_argument('-wordlength', action='store')
#     opt.add_argument('-detail', action='store_true')
#     opt.add_argument('-mode', action='store')
#     opt.add_argument('-triads_max_num', action='store')
#     opt.add_argument('-triads_overlap', action='store_true')
#     opt.add_argument('-cdump', action='store_true')
#     opt.add_argument('-configfile', action='store')
#     opt.add_argument('-man', action='store_true')
#     opt.add_argument('-debug', type=int, action='store', default=1)

#     if False:
#         options = parser.parse_args()
#     else: # TODO: only for debugging
#         # os.chdir(r'C:\Users\cleisonp\OneDrive - Alcast do Brasil SA\Documentos\GitHub\carpalx-py')
#         os.chdir(r'D:\Cleison\Documents\GitHub\carpalx-py')
#         options = parser.parse_args(r'-configfile etc/tutorial-00.ini -nocache'.split())

#     # pod2usage() if $OPT{help};
#     # if options.man:
#     #   pod2usage(-verbose=>2) if $OPT{man};
#     global config, keyboard_input, effort_model, effort_k_param, effort_weight_param
#     global effort_path_cost, effort_finger_distance, colors, kb_modes
#     config = load_configuration(options.configfile)
#     options.configfile = config.get('main', 'configfile')

#     keyboard_input = parse_conf_file(config.get('kb_definition', 'keyboard_input'))
#     effort_model = parse_conf_file(config.get('model', 'effort_model'))
#     effort_k_param = parse_conf_file(effort_model.get('main', 'k_param'))
#     effort_weight_param = parse_conf_file(effort_model.get('main', 'weight_param'))
#     effort_path_cost = parse_conf_file(effort_model.get('main', 'path_cost'))
#     effort_finger_distance = parse_conf_file(effort_model.get('main', 'finger_distance'))

#     colors = parse_conf_file(config.get('kb_parameters', 'colors'))
    
#     kb_modes = parse_conf_file(config.get('kb_parameters', 'modes'))

#     populate_configuration() # copy command line options to config hash
#     # TODO: only for debugging
#     config['options']['debug'] = '0'
#     # validateconfiguration();
#     # if($CONF{cdump}) {
#     #   $Data::Dumper::Pad = "debug parameters";
#     #   $Data::Dumper::Indent = 1;
#     #   $Data::Dumper::Quotekeys = 0;
#     #   $Data::Dumper::Terse = 1;
#     #   print Dumper(\%CONF);
#     #   exit;
#     # }

#     actions = config.get('main', 'action').split(',')
#     _effortlookup = {}

#     # my ($keytriads,$keyboard);

#     for action in actions:
#         print_debug(1, f'found action {action}')
#         match action:
#             case 'loadtriads':
#                 # read the document and extract triads
#                 #
#                 # triads are adjacent three-key combinations parsed from the
#                 # document based on the setting of the mode=MODE value
#                 # (see <mode MODE> block for filters for the MODE).
#                 #
#                 print_debug(1, 'loading triad from corpus file', config.get('corpus', 'corpus'))
#                 keytriads = read_document(config.get('corpus', 'corpus'))
#             case 'loadkeyboard':
#                 # create a keyboard and the associated effort matrix
#                 print_debug(1, 'loading keyboard from', config.get('kb_definition', 'keyboard_input'))
#                 keyboard = create_keyboard(config.get('kb_definition', 'keyboard_input'))
#             case 'reporttriads':
#                 print_debug(1, 'reporting triad frequency')
#                 self.report_triads(keytriads, keyboard)
#             case 'reportwordeffort':
#                 print_debug(1, 'reporting word efforts')
#                 report_word_effort(keyboard)
#             case 'drawinputkeyboard':
#                 print_debug(1, 'drawing input keyboard')
#                 draw_keyboard(keyboard, config.get('kb_parameters', 'pngfile_keyboard_input'), {'title': f'{config.get("kb_definition", "keyboard_input")} layout'})
#                 print_keyboard(keyboard)
#             case 'drawoutputkeyboard':
#                 print_debug(1, 'drawing output keyboard')
#                 draw_keyboard(keyboard, config.get('kb_parmeters', 'pngfile_keyboard_output'), {'title': f'Optimized layout, runid {config.get("main", "runid")}'})
#                 print_keyboard(keyboard)
#             case s if s.startswith('reporteffort'):
#                 # calculate the canonical effort associated with the original
#                 # keyboard layout - the layout will be altered to try to minimize this
#                 report_option = s.removeprefix('reporteffort')
#                 print_debug(1, 'calculating effort')
#                 config['options']['memorize'] = 'no'
#                 report_keyboard_effort(keytriads, keyboard, report_option)
#                 config['options']['memorize'] = 'yes'
#             case 'optimize':
#                 # optimize the keyboard layout to decrease the effort
#                 global mask_config
#                 mask_config = parse_conf_file(config.get('kb_parameters', 'mask'))
#                 timer_start = default_timer()
#                 keyboard = optimize_keyboard(keytriads, keyboard)
#                 timer_stop = default_timer()
#                 print_keyboard(keyboard)
#                 print('Total time spent optimizing:', '{:.3f}'.format(timer_stop-timer_start))
#             case 'exit':
#                 sys.exit()
#             case other:
#                 raise KeyError(f'cannot understand action {action}')
#     sys.exit()

# %% execution

if __name__ == '__main__':
    os.chdir(r'C:\\Users\\cleisonp\\OneDrive - Alcast do Brasil SA\\Documentos\\GitHub\\carpalx-py')
    # logger = log.get_logger(__name__, level=log.logging.INFO)
    run = Carpalx('etc//tutorial-00 copy.ini')
# %%
