#!/home/martink/bin/perl
# %% imports
import argparse
import configparser
import os
import re
import sys
from tqdm import tqdm

import json
from itertools import islice
from collections import defaultdict, Counter
from random import randint

# %% header

=pod

=head1 NAME

carpalx - given text input, determine optimal keyboard mapping to minimize typing effort based on a typing effort model

=head1 SYNOPSIS

  # all configuration read from etc/carpalx.conf
  carpalx -keyboard_input keyboard.conf -keyboard_output keyboard-optimized.conf
          -corpus corpus/words.txt
          -action loadkeyboard,loadtriads,optimize
          -conf etc/carpalx.conf
          [-debug]

=head1 DESCRIPTION

carpalx is a keyboard layout optimizer. Given a training corpus
(e.g. English text) and parameters that describe typing effort,
carpalx uses simulated annealing to find a keyboard layout to minimize
typing effort.

Typing effort is modeled using three contributions. First, base effort
is derived from finger travel distance. Second, row, hand and finger
penalties are added to limit use of weaker fingers/hands and
distinguish harder-to-reach keys. Third, stroke path effort is used to
rate the effort based on finger, row and hand alternation (e.g. asd is
much easier to type than sad).

=head1 CONFIGURATION

=head2 Configuration file name and path

carpalx will look in the following locations for a configuration file

  .
  SCRIPT_BIN/../etc
  SCRIPT_BIN/etc
  SCRIPT_BIN/
  ~/.carpalx/etc
  ~/.carpalx

where SCRIPT_BIN is the location of the carpalx script. If the name of
the configuration file is not passed via -conf, then SCRIPT_NAME.conf
is tried where SCRIPT_NAME is the name of the script. For example,

  > cd carpalx-0.11
  > bin/carpalx

will attempt to find carpalx.conf in the above paths.

Using -debug -debug will dump the configuration parameters.

  > bin/carpalx -debug -debug

=head2 Configuration structure

The configuration file comprises variable-value pairs, which may be
placed in blocks.

  a = 1
  <someblock>
    b = 2
    <anotherblock>
    c = 3
    </anotherblock>
  </someblock>

Combinations of related parameters (e.g. base effort, keyboard
configuration) are stored in individual files
(e.g. etc/mask/letters.conf) which are subsequently imported into
the main configuration file using <<include>>

  ...
  <<include etc/mask/letters.conf>>
  ...

=head1 HISTORY

=over

=item * 0.10

Packaged and versioned code.

=item * 0.11

Adjusted typing model to include weights for base, effort and stroke components.

Improved clarity of effort reports.

Improved consistency in configuration file.

Added fonts/

=item * 0.12

Can now load a cache file basedon parsed corpus instead of original corpus.

=back

=head1 BUGS

Report!

=head1 AUTHOR

Martin Krzywinski <martink@bcgsc.ca>
http://mkweb.bcgsc.ca

=head1 CONTACT

  Martin Krzywinski
  Genome Sciences Centre
  100-570 W 7th Ave
  Vancouver BC V5Z 4S6

=cut

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

use strict;
use Cwd;
use Config::General;
use Data::Dumper;
use File::Basename;
use File::Spec;
use FindBin;
use Getopt::Long;
use GD;
use IO::File;
use Math::VecStat qw(sum min max average);
use Set::IntSpan;
use Pod::Usage;
use Storable qw(store retrieve dclone);
use Digest::MD5 qw(md5_hex);
use Time::HiRes qw(gettimeofday tv_interval);
use lib "$FindBin::RealBin";
use lib "$FindBin::RealBin/../lib";
use lib "$FindBin::RealBin/lib";
use vars qw(%OPT %CONF);

################################################################
################################################################

# %% report_keyboard_effort
=pod

=head1 INTERNAL FUNCTIONS

The content below may be out of date

=cut

sub report_keyboard_effort {

  my ($keytriads,$keyboard,$option) = @_;

  my %effort;
  $effort{all}       = calculate_effort($keytriads,$keyboard);

  my %CONF_prev      = dclone(\%CONF);

  # recall triad effort is
  #
  # $kb * $k1*$be1 * ( 1 + $k2*$be2 * ( 1 + $k3*$be3 ) ) +
  # $kp * $k1*$pe1 * ( 1 + $k2*$pe2 * ( 1 + $k3*$pe3 ) ) +
  # $ks * $s
  #
  # be1,be2,be3 baseline efforts for first, second and third key in triad
  # be1,be2,be3 penalty efforts for first, second and third key in triad
  # s stroke path

  {
    # baseline effort
    # kp=ks=0
    local $CONF{effort_model}{k_param}{kp}                             = 0;
    local $CONF{effort_model}{k_param}{ks}                             = 0;
    my $keyboard_new   = create_keyboard($CONF{keyboard_input});
    $effort{base}    = calculate_effort($keytriads,$keyboard_new);
  }
  {
    # penalty effort
    # kb=ks=0
    local $CONF{effort_model}{k_param}{kb}                             = 0;
    local $CONF{effort_model}{k_param}{ks}                             = 0;
    my $keyboard_new = create_keyboard($CONF{keyboard_input});
    $effort{penalty}= calculate_effort($keytriads,$keyboard_new);
  }
  {
    # stroke effort
    # kb=kp=0
    local $CONF{effort_model}{k_param}{kb}                             = 0;
    local $CONF{effort_model}{k_param}{kp}                             = 0;
    my $keyboard_new = create_keyboard($CONF{keyboard_input});
    $effort{path}= calculate_effort($keytriads,$keyboard_new);
  }
  {
    # hand penalty only
    local $CONF{effort_model}{k_param}{kb}                             = 0;
    local $CONF{effort_model}{k_param}{ks}                             = 0;
    local $CONF{effort_model}{weight_param}{penalties}{default}        = 0;
    local $CONF{effort_model}{weight_param}{penalties}{weight}{row}    = 0;
    local $CONF{effort_model}{weight_param}{penalties}{weight}{finger} = 0;
    my $keyboard_new = create_keyboard($CONF{keyboard_input});
    $effort{penalty_hand}= calculate_effort($keytriads,$keyboard_new);
  }
  {
    # row penalty only
    local $CONF{effort_model}{k_param}{kb}                             = 0;
    local $CONF{effort_model}{k_param}{ks}                             = 0;
    local $CONF{effort_model}{weight_param}{penalties}{default}        = 0;
    local $CONF{effort_model}{weight_param}{penalties}{weight}{hand}   = 0;
    local $CONF{effort_model}{weight_param}{penalties}{weight}{finger} = 0;
    my $keyboard_new = create_keyboard($CONF{keyboard_input});
    $effort{penalty_row}= calculate_effort($keytriads,$keyboard_new);
  }
  {
    # finger penalty only
    local $CONF{effort_model}{k_param}{kb}                             = 0;
    local $CONF{effort_model}{k_param}{ks}                             = 0;
    local $CONF{effort_model}{weight_param}{penalties}{default}        = 0;
    local $CONF{effort_model}{weight_param}{penalties}{weight}{hand}   = 0;
    local $CONF{effort_model}{weight_param}{penalties}{weight}{row}    = 0;
    my $keyboard_new = create_keyboard($CONF{keyboard_input});
    $effort{penalty_finger}= calculate_effort($keytriads,$keyboard_new);
  }

  {
    # one-key effort
    local $CONF{effort_model}{k_param}{k2}                             = 0;
    local $CONF{effort_model}{k_param}{k3}                             = 0;
    local $CONF{effort_model}{k_param}{ks}                             = 0;
    my $keyboard_new = create_keyboard($CONF{keyboard_input});
    $effort{k1}= calculate_effort($keytriads,$keyboard_new);
  }
  {
    # two-key effort
    local $CONF{effort_model}{k_param}{k3}                             = 0;
    local $CONF{effort_model}{k_param}{ks}                             = 0;
    my $keyboard_new = create_keyboard($CONF{keyboard_input});
    $effort{k12}= calculate_effort($keytriads,$keyboard_new);
  }
  {
    # three-key effort
    local $CONF{effort_model}{k_param}{ks}                             = 0;
    my $keyboard_new = create_keyboard($CONF{keyboard_input});
    $effort{k123}= calculate_effort($keytriads,$keyboard_new);
  }

  printinfo("Keyboard effort");
  printinfo("-"x60);

  my %efforts = ( k1 => [ $effort{k1},
        100*sdiv($effort{k1},$effort{k123}),
        100*sdiv($effort{k1},$effort{k123})],
      k12 => [ $effort{k12},
         100*sdiv($effort{k12}-$effort{k1},$effort{k123}),
         100*sdiv($effort{k12},$effort{k123}) ],
      k123 => [ $effort{k123},
          100*sdiv($effort{k123}-$effort{k12},$effort{k123}),
          100*sdiv($effort{k123},$effort{k123})],
      base => [ $effort{base},
          100*sdiv($effort{base},$effort{all}),
          100*sdiv($effort{base},$effort{all}) ],
      penalty => [ $effort{penalty},
             100*sdiv($effort{penalty},$effort{all}),
             100*sdiv($effort{base}+$effort{penalty},$effort{penalty}) ],
      penalty_hand => [ $effort{penalty_hand},
             100*sdiv($effort{penalty_hand},$effort{penalty}),
             100*sdiv($effort{penalty_hand},$effort{penalty}) ],
      penalty_row => [ $effort{penalty_row},
             100*sdiv($effort{penalty_row},$effort{penalty}),
             100*sdiv($effort{penalty_hand}+$effort{penalty_row},$effort{penalty}) ],
      penalty_finger => [ $effort{penalty_finger},
             100*sdiv($effort{penalty_finger},$effort{penalty}),
             100*sdiv($effort{penalty_hand}+$effort{penalty_row}+$effort{penalty_finger},$effort{penalty}) ],
      path => [ $effort{path},
          100*sdiv($effort{path},$effort{all}),
          100*sdiv($effort{base}+$effort{penalty}+$effort{path},$effort{all}) ],
      all => [ $effort{all},
          100*sdiv($effort{all},$effort{all}),
          100*sdiv($effort{all},$effort{all}) ],
    );

  printinfo(sprintf("%-20s %8.3f %5.1f %5.1f","k1",
        @{$efforts{k1}}));
  printinfo(sprintf("%-20s %8.3f %5.1f %5.1f","k1,k2",
        @{$efforts{k12}}));
  printinfo(sprintf("%-20s %8.3f %5.1f %5.1f","k1,k2,k3",
        @{$efforts{k123}}));
  printinfo(sprintf("%-20s %8.3f %5.1f %5.1f","b",
        @{$efforts{base}}));
  printinfo(sprintf("%-20s %8.3f %5.1f %5.1f","p",
        @{$efforts{penalty}}));
  printinfo(sprintf("%-20s %8.3f %5.1f %5.1f","ph",
        @{$efforts{penalty_hand}}));
  printinfo(sprintf("%-20s %8.3f %5.1f %5.1f","pr",
        @{$efforts{penalty_row}}));
  printinfo(sprintf("%-20s %8.3f %5.1f %5.1f","pf",
        @{$efforts{penalty_finger}}));
  printinfo(sprintf("%-20s %8.3f %5.1f %5.1f","s",
        @{$efforts{path}}));
  printinfo(sprintf("%-20s %8.3f %5.1f %5.1f","all",
        @{$efforts{all}}));
  printinfo();
  for my $var (qw(k1 k12 k123 base penalty penalty_hand penalty_row penalty_finger path all)) {
    printinfo(sprintf("#data effort_%s=>[%.3f,%.3f,%.3f],",
          $var,
          @{$efforts{$var}}));
  }

  return if $option eq "verybrief";

  printinfo();

  my $stats;
  for my $triad (keys %$keytriads) {
    my $ntriad = $keytriads->{$triad};
    my $char   = substr($triad,0,1);
    my $row    = $keyboard->{map}{$char}{row};
    my $hand   = $keyboard->{map}{$char}{hand};
    my $finger = $keyboard->{map}{$char}{finger};
    $stats->{row}{$row} += $ntriad;
    $stats->{hand}{$hand} += $ntriad;
    $stats->{finger}{$finger} += $ntriad;
  }

  histogram($stats->{row},"keyboard row frequency","row");
  histogram($stats->{hand},"keyboard hand frequency","hand");
  histogram($stats->{finger},"keyboard finger frequency","finger");

  my $charlist = read_document($CONF{corpus},{charlist=>1});
  my $runlength;

  printdebug(1,"calculating runs");
  for my $i (0..@$charlist-1) {
    $stats->{charfreq}{$charlist->[$i]}++;
    if($i==0) {
      $runlength->{rowjump} = 1;
      $runlength->{finger}  = 1;
      $runlength->{hand}    = 1;
      $runlength->{row}     = 1;
    } else {
      $stats->{charpairfreq}{ $charlist->[$i-1] . $charlist->[$i] }++;
      my $h1 = $keyboard->{map}{$charlist->[$i]}{hand};
      my $h2 = $keyboard->{map}{$charlist->[$i-1]}{hand};
      my $r1 = $keyboard->{map}{$charlist->[$i]}{row};
      my $r2 = $keyboard->{map}{$charlist->[$i-1]}{row};
      if ($h1 == $h2 && $r1 != $r2) {
        $runlength->{rowjump} += abs($r1-$r2);
      } else {
        $stats->{run}{rowjump}{$runlength->{rowjump}}++;
        $runlength->{rowjump} = 1;
      }
      if($i == @$charlist-1) {
        $stats->{run}{rowjump}{$runlength->{rowjump}}++;
      }
      for my $runtype (qw(finger hand row)) {
        my $cv  = $keyboard->{map}{$charlist->[$i]}{$runtype};
        my $cvp = $keyboard->{map}{$charlist->[$i-1]}{$runtype};
        if ($cv == $cvp) {
          $runlength->{$runtype}++;
          } else {
            $stats->{run}{$runtype}{all}{ $runlength->{$runtype} }++;
            if($runtype ne "finger") {
              $stats->{run}{$runtype}{ $cvp }{ $runlength->{$runtype} }++;
            }
            $runlength->{$runtype} = 1;
          }
        if($i == @$charlist-1) {
          $stats->{run}{$runtype}{all}{ $runlength->{$runtype} }++;
          $stats->{run}{$runtype}{ $cvp }{ $runlength->{$runtype} }++ if $runtype ne "finger";
        }
      }
    }
  }
  histogram($stats->{run}{hand}{0},"keyboard left hand run length","left_hand_run");
  histogram($stats->{run}{hand}{1},"keyboard right hand run length","right_hand_run");
  histogram($stats->{run}{hand}{all},"keyboard hand run length","all_hand_run");
  histogram($stats->{run}{row}{1},"keyboard top row run length","t_row_run");
  histogram($stats->{run}{row}{2},"keyboard home row run length","h_row_run");
  histogram($stats->{run}{row}{3},"keyboard bottom row run length","b_row_run");
  histogram($stats->{run}{row}{all},"keyboard row run length","all_row_run");
  histogram($stats->{run}{finger}{all},"keyboard finger run length","finger_run");
  histogram($stats->{run}{rowjump},"keyboard same-hand row jump length","row_jump");
  histogram($stats->{charfreq},"corpus character frequency","character_frequency","value");
  histogram($stats->{charpairfreq},"corpus character pair frequency","character_pair_frequency","value");
}

def report_keyboard_effort(keytriads, keyboard, option):

    effort = {}
    effort['all'] = calculate_effort(keytriads, keyboard)
    import copy

    # recall: triad effort is
    #
    # kb * k1*be1 * ( 1 + k2*be2 * ( 1 + k3*be3 ) ) +
    # kp * k1*pe1 * ( 1 + k2*pe2 * ( 1 + k3*pe3 ) ) +
    # ks * s
    #
    # be1,be2,be3 baseline efforts for first, second and third key in triad
    # pe1,pe2,pe3 penalty efforts for first, second and third key in triad
    # s stroke path

    local_k = copy.deepcopy(effort_k_param)
    # baseline effort
    # kp=ks=0
    local_k['effort']['kp'] = '0'
    local_k['effort']['ks'] = '0'

    keyboard_new = create_keyboard(config.get('kb_definition', 'keyboard_input'))
    effort['base'] = calculate_effort(keytriads, keyboard_new, local_k)

    local_k = copy.deepcopy(effort_k_param)
    # penalty effort
    # kb=ks=0
    local_k['effort']['kb'] = '0'
    local_k['effort']['ks'] = '0'
    keyboard_new = create_keyboard(config.get('kb_definition', 'keyboard_input'))
    effort['penalty'] = calculate_effort(keytriads, keyboard_new, local_k)

    local_k = copy.deepcopy(effort_k_param)
    # stroke effort
    # kb=kp=0
    local_k['effort']['kb'] = '0'
    local_k['effort']['kp'] = '0'
    keyboard_new = create_keyboard(config.get('kb_definition', 'keyboard_input'))
    effort['path'] = calculate_effort(keytriads, keyboard_new, local_k)

    local_k = copy.deepcopy(effort_k_param)
    local_weight = copy.deepcopy(effort_weight_param)
    # hand penalty only
    local_k['effort']['kb'] = '0'
    local_k['effort']['ks'] = '0'
    local_weight['main']['default'] = '0'
    local_weight['weight']['row'] = '0'
    local_weight['weight']['finger'] = '0'
    keyboard_new = create_keyboard(config.get('kb_definition', 'keyboard_input'))
    effort['penalty_hand'] = calculate_effort(keytriads, keyboard_new, local_k, local_weight)

    local_k = copy.deepcopy(effort_k_param)
    local_weight = copy.deepcopy(effort_weight_param)
    # row penalty only
    local_k['effort']['kb'] = '0'
    local_k['effort']['ks'] = '0'
    local_weight['main']['default'] = '0'
    local_weight['weight']['hand'] = '0'
    local_weight['weight']['finger'] = '0'
    keyboard_new = create_keyboard(config.get('kb_definition', 'keyboard_input'))
    effort['penalty_row'] = calculate_effort(keytriads, keyboard_new, local_k, local_weight)

    local_k = copy.deepcopy(effort_k_param)
    local_weight = copy.deepcopy(effort_weight_param)
    # finger penalty only
    local_k['effort']['kb'] = '0'
    local_k['effort']['ks'] = '0'
    local_weight['main']['default'] = '0'
    local_weight['weight']['hand'] = '0'
    local_weight['weight']['row'] = '0'
    keyboard_new = create_keyboard(config.get('kb_definition', 'keyboard_input'))
    effort['penalty_finger'] = calculate_effort(keytriads, keyboard_new, local_k, local_weight)

    local_k = copy.deepcopy(effort_k_param)
    local_weight = copy.deepcopy(effort_weight_param)
    # one-key effort
    local_k['effort']['k2'] = '0'
    local_k['effort']['k3'] = '0'
    local_k['effort']['ks'] = '0'
    keyboard_new = create_keyboard(config.get('kb_definition', 'keyboard_input'))
    effort['k1'] = calculate_effort(keytriads, keyboard_new, local_k)

    local_k = copy.deepcopy(effort_k_param)
    # two-key effort
    local_k['effort']['k3'] = '0'
    local_k['effort']['ks'] = '0'
    keyboard_new = create_keyboard(config.get('kb_definition', 'keyboard_input'))
    effort['k12'] = calculate_effort(keytriads, keyboard_new, local_k)

    local_k = copy.deepcopy(effort_k_param)
    # three-key effort
    local_k['effort']['ks'] = '0'
    keyboard_new = create_keyboard(config.get('kb_definition', 'keyboard_input'))
    effort['k123'] = calculate_effort(keytriads, keyboard_new, local_k)

    efforts = {'k1': [effort['k1'],
                      100*(effort['k1']/effort['k123']),
                      100*(effort['k1']/effort['k123'])],
               'k12': [effort['k12'],
                       100*((effort['k12']-effort['k1'])/effort['k123']),
                       100*(effort['k12']/effort['k123'])],
               'k123': [effort['k123'],
                        100*((effort['k123']-effort['k12'])/effort['k123']),
                        100*(effort['k123']/effort['k123'])],
               'base': [effort['base'],
                        100*(effort['base']/effort['all']),
                        100*(effort['base']/effort['all'])],
               'penalty': [effort['penalty'],
                           100*(effort['penalty']/effort['all']),
                           100*((effort['base']+effort['penalty'])/effort['penalty'])],
               'penalty_hand': [effort['penalty_hand'],
                                100*(effort['penalty_hand']/effort['penalty']),
                                100*(effort['penalty_hand']/effort['penalty'])],
               'penalty_row': [effort['penalty_row'],
                               100*(effort['penalty_row']/effort['penalty']),
                               100*((effort['penalty_hand']+effort['penalty_row'])/effort['penalty'])],
               'penalty_finger': [effort['penalty_finger'],
                                  100*(effort['penalty_finger']/effort['penalty']),
                                  100*((effort['penalty_hand']+effort['penalty_row']+effort['penalty_finger'])/effort['penalty'])],
               'path': [effort['path'],
                        100*(effort['path']/effort['all']),
                        100*((effort['base']+effort['penalty']+effort['path'])/effort['all'])],
               'all': [effort['all'],
                       100*(effort['all']/effort['all']),
                       100*(effort['all']/effort['all'])]}

    print("Keyboard effort")
    print("-"x60)

    print('k1\t\t', '{:5.3f}{:6.1f}{:6.1f}'.format(*efforts['k1']))
    print('k1,k2\t\t', '{:5.3f}{:6.1f}{:6.1f}'.format(*efforts['k12']))
    print('k1,k2,k3\t', '{:5.3f}{:6.1f}{:6.1f}'.format(*efforts['k123']))
    print('b\t\t', '{:5.3f}{:6.1f}{:6.1f}'.format(*efforts['base']))
    print('p\t\t', '{:5.3f}{:6.1f}{:6.1f}'.format(*efforts['penalty']))
    print('ph\t\t', '{:5.3f}{:6.1f}{:6.1f}'.format(*efforts['penalty_hand']))
    print('pr\t\t', '{:5.3f}{:6.1f}{:6.1f}'.format(*efforts['penalty_row']))
    print('pf\t\t', '{:5.3f}{:6.1f}{:6.1f}'.format(*efforts['penalty_finger']))
    print('s\t\t', '{:5.3f}{:6.1f}{:6.1f}'.format(*efforts['path']))
    print('all\t\t', '{:5.3f}{:6.1f}{:6.1f}'.format(*efforts['all']))
    print('\n')

    for var in ['k1', 'k12', 'k123', 'base', 'penalty', 'penalty_hand', 'penalty_row',
                'penalty_finger', 'path', 'all']:
        print(f'#data effort_{var}=>', '{:5.3f},{:6.3f},{:6.3f}'.format(*efforts[var]))

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
        row    = keyboard['map'][char]['row']
        hand   = keyboard['map'][char]['hand']
        finger = keyboard['map'][char]['finger']
        stats['row'][row] += ntriad
        stats['hand'][hand] += ntriad
        stats['finger'][finger] += ntriad

    # TODO: review this function
    # histogram($stats->{row},"keyboard row frequency","row");
    # histogram($stats->{hand},"keyboard hand frequency","hand");
    # histogram($stats->{finger},"keyboard finger frequency","finger");

    charlist = read_document(config.get('corpus', 'corpus'), charlist=keytriads)
    # my $runlength;

    print_debug(1, "calculating runs")
    for my $i (0..@$charlist-1) {
        $stats->{charfreq}{$charlist->[$i]}++;
        if($i==0) {
        $runlength->{rowjump} = 1;
        $runlength->{finger}  = 1;
        $runlength->{hand}    = 1;
        $runlength->{row}     = 1;
        } else {
        $stats->{charpairfreq}{ $charlist->[$i-1] . $charlist->[$i] }++;
        my $h1 = $keyboard->{map}{$charlist->[$i]}{hand};
        my $h2 = $keyboard->{map}{$charlist->[$i-1]}{hand};
        my $r1 = $keyboard->{map}{$charlist->[$i]}{row};
        my $r2 = $keyboard->{map}{$charlist->[$i-1]}{row};
        if ($h1 == $h2 && $r1 != $r2) {
            $runlength->{rowjump} += abs($r1-$r2);
        } else {
            $stats->{run}{rowjump}{$runlength->{rowjump}}++;
            $runlength->{rowjump} = 1;
        }
        if($i == @$charlist-1) {
            $stats->{run}{rowjump}{$runlength->{rowjump}}++;
        }
        for my $runtype (qw(finger hand row)) {
            my $cv  = $keyboard->{map}{$charlist->[$i]}{$runtype};
            my $cvp = $keyboard->{map}{$charlist->[$i-1]}{$runtype};
            if ($cv == $cvp) {
            $runlength->{$runtype}++;
            } else {
                $stats->{run}{$runtype}{all}{ $runlength->{$runtype} }++;
                if($runtype ne "finger") {
                $stats->{run}{$runtype}{ $cvp }{ $runlength->{$runtype} }++;
                }
                $runlength->{$runtype} = 1;
            }
            if($i == @$charlist-1) {
            $stats->{run}{$runtype}{all}{ $runlength->{$runtype} }++;
            $stats->{run}{$runtype}{ $cvp }{ $runlength->{$runtype} }++ if $runtype ne "finger";
            }
        }
        }
    }
    histogram($stats->{run}{hand}{0},"keyboard left hand run length","left_hand_run");
    histogram($stats->{run}{hand}{1},"keyboard right hand run length","right_hand_run");
    histogram($stats->{run}{hand}{all},"keyboard hand run length","all_hand_run");
    histogram($stats->{run}{row}{1},"keyboard top row run length","t_row_run");
    histogram($stats->{run}{row}{2},"keyboard home row run length","h_row_run");
    histogram($stats->{run}{row}{3},"keyboard bottom row run length","b_row_run");
    histogram($stats->{run}{row}{all},"keyboard row run length","all_row_run");
    histogram($stats->{run}{finger}{all},"keyboard finger run length","finger_run");
    histogram($stats->{run}{rowjump},"keyboard same-hand row jump length","row_jump");
    histogram($stats->{charfreq},"corpus character frequency","character_frequency","value");
    histogram($stats->{charpairfreq},"corpus character pair frequency","character_pair_frequency","value");
    }

# %% sdiv

sub sdiv {
  my ($x,$y) = @_;
  return $y ? $x/$y : 0;
}

# %% histogram

sub histogram {
  my $table = shift;
  my $title = shift;
  my $datatitle = shift;
  my $sortfunc = shift || "num";
  my @values;
  if($sortfunc eq "num") {
    @values = sort {$a <=> $b} keys %$table;
  } elsif ($sortfunc eq "ascii") {
    @values = sort {$a cmp $b} keys %$table;
  } elsif ($sortfunc eq "value") {
    @values = sort {$table->{$b} <=> $table->{$a}} keys %$table;
  }
  my $total  = sum ( map {$table->{$_}} @values );
  my $running_total = 0;
  my $data_table;
  if($title) {
    printinfo($title);
    printinfo("-"x60);
  }
  for my $value (@values) {
    $running_total += $table->{$value};
    push @{$data_table->{data}}, $value || 0;
    push @{$data_table->{freq}}, $table->{$value} / $total;
    push @{$data_table->{cumul}}, $running_total / $total;
    printinfo(sprintf("%-20s %8d %4.1f %5.1f",
          $value,
          $table->{$value},
          100*$table->{$value}/$total,
          100*$running_total/$total));
  }
  printinfo();
  printinfo(sprintf("#data %s_data=>[qw(%s)],",$datatitle,join(" ",@{$data_table->{data}})));
  printinfo(sprintf("#data %s_frequency=>[%s],",$datatitle,join(",",map { sprintf("%.3f",$_) } @{$data_table->{freq}})));
  printinfo(sprintf("#data %s_cumulative=>[%s],",$datatitle,join(",",map { sprintf("%.3f",$_) } @{$data_table->{cumul}})));
  printinfo();
}

# %% find_action

# report the frequency and cumulative frequency of all triads

sub find_action {
  my ($rx,@actions) = @_;
  if ( my ($action) = grep($rx,@actions) ) {
    return $action;
  } else {
    return undef;
  }
}

# %% advance_actions

sub advance_actions {
  my ($action,@actions) = @_;
  exit if $action =~ /exit|quit/;
  my @newactions;
  my $found;
  for $a (@actions) {
    if(! $found && $a eq $action) {
      $found = 1;
      next;
    }
    push @newactions, $a;
  }
  return @newactions;
}

# %% report_triads

# sub report_triads {
#   my ($keytriads,$keyboard) = @_;
#   my $n = sum(values %$keytriads);
#   my $nc = 0;
#   for my $triad (sort {$keytriads->{$b} <=> $keytriads->{$a}} keys %$keytriads) {
#     $nc += $keytriads->{$triad};
#     my $effort = $keyboard ? calculate_triad_effort($triad,$keyboard,@{$CONF{effort_model}{k_param}}{qw(k1 k2 k3 kb kp ks)}) : "na";
#     printinfo("triad",$triad,$keytriads->{$triad},$keytriads->{$triad}/$n,$nc/$n,"effort",$effort);
#   }
# }

def report_triads(keytriads, keyboard=None):
    n = sum(keytriads.values())
    nc = 0
    for triad, freq in keytriads.items():
        nc += freq
        effort = calculate_triad_effort(triad, keyboard, dict(effort_k_param['effort'])) if keyboard is not None else None
        print('triad', triad, freq, freq/n, nc/n, 'effort', effort)

# %% resolve_path

# sub resolve_path {
#   my $file = shift;
# 	my $is_absolute = File::Spec->file_name_is_absolute( $file );
#   if($is_absolute) {
#     return $file;
#   } else {
#     return "$CONF{configdir}/$file";
#   }
# }

def resolve_path(file):
    if os.path.isabs(file):
        return file
    else:
        return os.path.join(config.get('main', 'configdir'), file)

# %% report_word_effort

# sub report_word_effort {
#     my $keyboard = shift;
#     open(WORDS, $CONF{words} =~ /^\// ? $CONF{words} : $FindBin::RealBin . "/$CONF{words}") || die "cannot open word list file $CONF{words}";
#     my @words = <WORDS>;
#     chomp @words;
#     close(WORDS);
#     if($CONF{wordlength}) {
#       my $length = Set::IntSpan->new($CONF{wordlength});
#       @words = grep($length->member(length($_)), @words);
#     }
#     #@words = grep($_ =~ /^[a-z]+$/, map { lc $_ } @words);
#     my $wordeffort = rankwords(\@words,$keyboard);
#     summarizerankwords($wordeffort);
# }

def report_word_effort(keyboard):
    word_file = resolve_path(config.get('wordstats', 'words'))
    if not os.path.exists(word_file):
        raise FileNotFoundError(f'cannot find word list file {word_file}')
    with open(word_file, 'r') as f:
        words = f.read().splitlines()
    if config.getint('options', 'debug') > 1 and len(words) > 100_000:
        words = words[:100_000]
    if config.get('wordstats', 'wordlength'):
        min_len, max_len = config.get('wordstats', 'wordlength').split('-')
        min_len, max_len = int(min_len), int(max_len)
        for word in tqdm(words.copy(), desc='parsing words'):
            if len(word) < min_len or len(word) > max_len:
                words.remove(word)
    word_effort = rank_words(words, keyboard)
    summarize_rank_words(word_effort)

# %% rank_words

# sub rankwords {
#   my $words = shift;
#   my $keyboard = shift;
#   my $wordeffort;
#   foreach my $word (@$words) {
#     my $wordtriads;
#     while($word =~ /(...)/g) {
#       my $triad = lc $1;
#       $wordtriads->{$triad}++;
#       pos $word -= 2;
#     }
#     next unless keys %$wordtriads;
#     my $word_effort = calculate_effort($wordtriads,$keyboard);
#     $wordeffort->{lc $word} = $word_effort;
#     printdebug(1,"wordeffort",lc $word,$word_effort);
#   }
#   return $wordeffort;
# }

def rank_words(words, keyboard):
    '''
    given a list of words, return the words and associated
    efforts, as calculated using the triads of the words.
    '''
    word_effort = {}
    for word in tqdm(words, desc='ranking words'):
        triads = [word[idx:idx+3].lower() for idx, item in enumerate(word)]
        triads = [triad for triad in triads if len(triad) == 3]
        word_triads = Counter(triads)
        word_effort[word] = calculate_effort(word_triads, keyboard)
        print_debug(1, 'word_effort', word, word_effort[word])
    return word_effort

# %% summarize_rank_words

# sub summarizerankwords {
#   my $wordeffort = shift;
#   my $topN = 25;
#   # top 10
#   my @sorted       = (sort {$wordeffort->{$b} <=> $wordeffort->{$a}} keys %$wordeffort);
#   if($CONF{detail}) {
#     for my $word (keys %$wordeffort) {
#       printinfo("wordeffort",$word,$wordeffort->{$word});
#     }
#   }
#   my @top10hardest = @sorted[0..$topN-1];
#   my @top10easiest = @sorted[@sorted-1-$topN..@sorted-1];
#   printinfo("wordreport","top $topN hardest",join(" ",map {sprintf ("%s:%4.1f",$_,$wordeffort->{$_})} @top10hardest));
#   printinfo("wordreport","top $topN easiest",join(" ",map {sprintf ("%s:%4.1f",$_,$wordeffort->{$_})} @top10easiest));
#   # percentiles
#   my $groups = 10;
#   for my $word (@top10hardest) {
#     printinfo("wordreport group",0,$word,$wordeffort->{$word});
#   }
#   foreach my $idx (0..$groups-1) {
#     my $elemidx = int($idx*@sorted/$groups);
#     my @words = @sorted[$elemidx..$elemidx+10];
#     my $cost = $wordeffort->{$words[0]};
#     printf ("wordreport percentile %d cost %.1f\n",int(100*$idx/$groups),$cost);
#     #print join(" ",join(" ",@words));
#     #print "\n";
#     for my $word (@words) {
#       printinfo("wordreport group",$idx+1,$word,$wordeffort->{$word});
#     }
#   }
#   for my $word (@top10easiest) {
#     printinfo("wordreport group",$groups+1,$word,$wordeffort->{$word});
#   }
# }

def summarize_rank_words(word_effort):
    '''
    Produce a summary of the word effort statistics.
    '''
    topN = 25
    # top 10
    sorted_desc = sorted(word_effort.items(), key=lambda item: item[1], reverse=True)
    sorted_asc = sorted(word_effort.items(), key=lambda item: item[1], reverse=False)
    if config.has_option('options', 'detail'):
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

# %% optimize_keyboard

=pod

=head2 optimize_keyboard()

  $newkeyboard = optimize_keyboard($keytriads,$keyboard);

Simulated annealing is used to search for a better keyboard layout. The function uses the list of triads, generated from the input text document, and an initial keyboard layout.

=cut

sub optimize_keyboard {
  die "more arguments needed in optimize_keyboard" unless @_ == 2;
  my ($keytriads,$keyboard) = @_;
  my $effort       = calculate_effort($keytriads,$keyboard);
  my $iterations   = $CONF{annealing}{iterations} || 1000;
  my ($t0,$k)      = @{$CONF{annealing}}{qw(t0 k)};
  # load up the mask - eligible keys for relocation
  my $mask         = _parse_mask($CONF{maskfilename});
  die "cannot create mask" unless $mask;
  # create a list of all keys that can be relocated
  my $reloc_list   = make_relocatable_list($mask);
  my $update_count = 0;
  my $last_reported_effort;

  my $keyboard_original = dclone($keyboard);
  my %seen_digests;
  for my $iter (1..$iterations) {
    my $time = [gettimeofday];
    my $keyboardnew;

    my $swap_range = $CONF{annealing}{maxswaps} - $CONF{annealing}{minswaps};
    my $swap_num   = $CONF{annealing}{minswaps};

    $swap_num      += int(rand($swap_range+1)) if $swap_range;
    if($CONF{annealing}{onestep}) {
      $keyboardnew     = _swap_keys($keyboard_original,$reloc_list,$swap_num);
      $effort          = calculate_effort($keytriads,$keyboard_original);
      my $digest = keyboard_digest($keyboardnew);
      if($seen_digests{$digest}) {
  # already seen this layout - fetch next layout
  next;
      }
      $seen_digests{$digest}++;
    } else {
      $keyboardnew     = _swap_keys($keyboard,$reloc_list,$swap_num);
    }
    # this is breaking???
    my $effortnew       = calculate_effort($keytriads,$keyboardnew);
    my $deffort         = $effortnew - $effort;
    my %report;
    $report{effort}    = $effort;
    $report{neweffort} = $effortnew;
    $report{deffort}   = $deffort;
    my $t = $t0*exp(-$iter*$k/$iterations);
    my $p = $CONF{annealing}{p0} * exp(-abs($deffort)/$t);
    $p = 1 if $p > 1; # float round-off
    $report{t} = $t;
    $report{p} = $p;

    my $keyboard_is_updated = 0;

    if( ($CONF{annealing}{action} eq "minimize" && $deffort < 0) ||
  ($CONF{annealing}{action} eq "maximize" && $deffort > 0) ) {
      # always accept layouts for which the effort is lower/higher (as prescribed by action)
      $effort   = $effortnew;
      $keyboard = $keyboardnew;
      $report{move} = "better/accept";
      $keyboard_is_updated = 1;
    } else {
      # sometimes accept layouts for which the effort is higher/lower (as prescribed by action)
      if(rand() < $p) {
  $report{move} = "worse/accept";
  $effort = $effortnew;
  $keyboard = $keyboardnew;
  $keyboard_is_updated = 1;
      } else {
  $report{move} = "worse/reject";
      }
    }
    $update_count += $keyboard_is_updated;

    my $make_report;
    if($CONF{report_filter} eq "all") {
      $make_report = 1;
    } elsif ($CONF{report_filter} eq "update") {
      $make_report = 1 if $report{move} =~ /accept/;
    } elsif ($CONF{report_filter} eq "lower") {
      $make_report = 1 if $deffort < 0;
    } elsif ($CONF{report_filter} eq "higher") {
      $make_report = 1 if $deffort > 0;
    } elsif ($CONF{report_filter} eq "lower_monotonic") {
      $make_report = 1 if ! defined $last_reported_effort || $effortnew < $last_reported_effort;
    } elsif ($CONF{report_filter} eq "higher_monotonic") {
      $make_report = 1 if ! defined $last_reported_effort || $effortnew > $last_reported_effort;
    }

    $report{move} .= "/report" if $make_report;

    my $make_draw;
    if($CONF{draw_filter} eq "all") {
      $make_draw = 1;
    } elsif ($CONF{draw_filter} eq "update") {
      $make_draw = 1 if $report{move} =~ /accept/;
    } elsif ($CONF{draw_filter} eq "lower") {
      $make_draw = 1 if $deffort < 0;
    } elsif ($CONF{draw_filter} eq "higher") {
      $make_draw = 1 if $deffort > 0;
    } elsif ($CONF{draw_filter} eq "lower_monotonic") {
      $make_draw = 1 if ! defined $last_reported_effort || $effortnew < $last_reported_effort;
    } elsif ($CONF{draw_filter} eq "higher_monotonic") {
      $make_draw = 1 if ! defined $last_reported_effort || $effortnew > $last_reported_effort;
    }

    my $stdout_report;
    if($CONF{stdout_filter} eq "all") {
      $stdout_report = 1;
    } elsif ($CONF{stdout_filter} eq "update") {
      $stdout_report = 1 if $report{move} =~ /accept/;
    } elsif ($CONF{stdout_filter} eq "lower") {
      $stdout_report = 1 if $deffort < 0;
    } elsif ($CONF{stdout_filter} eq "higher") {
      $stdout_report = 1 if $deffort > 0;
    } elsif ($CONF{stdout_filter} eq "lower_monotonic") {
      $stdout_report = 1 if ! defined $last_reported_effort || $effortnew < $last_reported_effort;
    } elsif ($CONF{stdout_filter} eq "higher_monotonic") {
      $stdout_report = 1 if ! defined $last_reported_effort || $effortnew > $last_reported_effort;
    }

    $report{move} .= "/draw" if $make_draw;

    my $parameters = {t=>$t,iter=>$iter,update_count=>$update_count,effort=>$effortnew,deffort=>$deffort};

    my $elapsed = tv_interval($time);

    # output to STDOUT is always lower monotonic
    if($stdout_report) {
      printkeyboard($keyboardnew);
      printf ("iter %6d effort %8.6f -> %8.6f d %10.8f p %10.8f t %10.8f %s cpu %s\n",
        $iter,@report{qw(effort neweffort deffort p t move)},$elapsed);
    }

    if($make_report && not $update_count % $CONF{report_period}) {
      report_keyboard($keyboard,$CONF{keyboard_output},$parameters);
      $last_reported_effort = $effortnew;
    }

    if($make_draw && not $update_count % $CONF{draw_period}) {
      draw_keyboard($keyboard,"$CONF{pngfile_keyboard_output}",$parameters);
      $last_reported_effort = $effortnew;
    }
  }
  return $keyboard;
}

# %% keyboard_digest

sub keyboard_digest {
  my $keyboard = shift;
  my @keys;
  for my $row (0..@{$keyboard->{key}}-1) {
    for my $col (0..@{$keyboard->{key}[$row]}-1) {
      push @keys, join("",@{$keyboard->{key}[$row][$col]}{qw(lc uc)});
    }
  }
  my $string = join(":",@keys);
  return md5_hex($string);
}

# %% report_keyboard

sub report_keyboard {
  my ($keyboard,$file,$parameters) = @_;
  $file = resolve_path($file);
  open(F,">$file");
  if($CONF{keyboard_output_show_parameters} =~ /current/) {
    print F "<current_parameters>\n";
    for my $parameter (keys %$parameters) {
      printf F ("%-18s = %s\n",$parameter,$parameters->{$parameter});
    }
    print F "</current_parameters>\n\n";
  }
  if($CONF{keyboard_output_show_parameters} =~ /annealing/) {
    print F "<annealing_parameters>\n";
    for my $parameter (keys %{$CONF{annealing}}) {
      printf F ("%-18s = %s\n",$parameter,$CONF{annealing}{$parameter});
    }
    print F "</annealing_parameters>\n\n";
  }

  print F "<keyboard>\n";
  for my $row (0..@{$keyboard->{key}}-1) {
    print F "<row ".($row+1).">\n";
    my (@keys,@fingers);
    for my $col (0..@{$keyboard->{key}[$row]}-1) {
      my $keystring;
      my ($lc,$uc) = @{$keyboard->{key}[$row][$col]}{qw(lc uc)};
      if($lc =~ /[a-z]/ && $uc eq uc $lc) {
  push @keys, $lc;
      } else {
  $lc = "\\".$lc if $lc eq "#";
  $uc = "\\".$uc if $uc eq "#";
  push @keys, "$lc$uc";
      }
      push @fingers, $keyboard->{key}[$row][$col]{finger};
    }
    printf F ("keys = %s\n",join(" ",@keys));
    printf F ("fingers  = %s\n",join(" ",@fingers));
    print F "</row>\n";
  }
  print F "</keyboard>\n\n";
  close(F);
}

# %% _swap_keys

=pod

=head2 _swap_keys()

  $newkeyboard = _swap_keys($keyboard,$reloc_list,$n);

Swap one or more pairs ($n randomly sampled pairs) of keys on the keyboard. Lower and upper case characters remain on the same key (e.g. no matter where 'a' is, A is always shift+a). This applies to both letter and non-letter characters (e.g. 1 and ! are always on the same key).

This function returns a new keyboard object with the keys swapped.

=cut

sub _swap_keys {
  my ($keyboard,$reloc_list,$n) = @_;
  my $keyboardcopy = dclone($keyboard);
  $n = 1 if ! $n;
  my $reloc_listsize = @$reloc_list;
  foreach (1..$n) {
    # pick two random keyboard locations from the list of relocatable keys
    my ($key1,$key2);
    while ( $key1 == $key2) {
      $key1 = $reloc_list->[rand($reloc_listsize)];
      $key2 = $reloc_list->[rand($reloc_listsize)];
    }
    # swap these two keys
    _swap_key_pair($keyboardcopy,@$key1,@$key2);
  }
  return $keyboardcopy;
}

# %% _swap_key_pair

=pod

=head2 _swap_key_pair()

  $key1 = [$row1,$col1];
  $key2 = [$row2,$col2];
  _swap_key_pair($keyboard,@$key1,@$key2);

This function modifies $keyboard in place.

=cut

sub _swap_key_pair {
  my ($keyboard,$row1,$col1,$row2,$col2) = @_;

  my ($k1lc,$k1uc) = @{ $keyboard->{key}[$row1][$col1] }{qw(lc uc)};
  my ($k2lc,$k2uc) = @{ $keyboard->{key}[$row2][$col2] }{qw(lc uc)};

  @{$keyboard->{key}[$row1][$col1]}{qw(lc uc)} = ($k2lc,$k2uc);
  @{$keyboard->{key}[$row2][$col2]}{qw(lc uc)} = ($k1lc,$k1uc);

  @{$keyboard->{map}}{$k1lc,$k1uc,$k2lc,$k2uc} = @{$keyboard->{map}}{$k2lc,$k2uc,$k1lc,$k1uc};
}

# %% calculate_effort

# sub calculate_effort {
#   my ($keytriads,$keyboard) = @_;
#   die "keyboard not defined " unless $keyboard;
#   die "triads not defined" unless $keytriads;
#   my $totaleffort = 0;
#   my $contributing_triads = 0;
#   my @k = @{$CONF{effort_model}{k_param}}{qw(k1 k2 k3 kb kp ks)};
#   foreach my $triad (keys %$keytriads) {
#     my $triad_effort     = calculate_triad_effort($triad,$keyboard,@k);
#     my $num_triads       = $keytriads->{$triad};
#     $totaleffort         += $triad_effort * $num_triads ;
#     $contributing_triads += $num_triads;
#     if($triad =~ /[*]/) {
#       #printinfo("calculate_effort",$triad,$num_triads,$triad_effort,$totaleffort,$contributing_triads);
#     }
#   }
#   $totaleffort /= $contributing_triads;
#   #printinfo("calculate_effort_done",$totaleffort);
#   #printinfo("*"x80);
#   return $totaleffort;
# }

def calculate_effort(keytriads, keyboard, k_params=effort_k_param, weight_params=effort_weight_param):
    '''
    Given a list of triads and the effort matrix, calculate the total carpal effort required to type the document from which the triads were generated. The effort is a non-negative number. The effort is a sum of the efforts for each triad. The total effort is normalized by the number of triads to remove dependency on document size.

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

    The form of this expression is motivated by the fact that the effort of three keystrokes is dependent on not only the individual identity of the keys but also alternation of hand, finger, row and column within the triad as well as presence of hard-to-type key combinations (e.g. zxc zqz awz ). For example, it is much easier to type "ttt" than "tbt", since the left forefinger must travel quite a distance in the latter example. Thus the insertion of the "b" character should impact the effort.

    In the first-order approximation k2=k3=k4=0 and the effort is simply the effort of typing the first key, effort(x). The individual effort of a key is defined in the <effort_row> blocks and is optionally modified by (a) shift penalty - CAPS are penalized and (b) hand penalty (e.g. you favour typing with your left hand). Since triads overlap, the first-order approximation for the entire document is the sum of the individual key efforts, without any long-range correlations.

    The addition of parameters k2 and k3 is designed to raise the effort of repeated difficult-to-type characters. This is where the notion of a triad comes into play. Notice that if effort(x) is zero, then the whole triad effort is zero.

    The patheffort(x,y,z) is a penalty which makes less desirable triads in which the keys do not follow a monotonic progression of columns, or triads which do not alternate hands. Once you try to type 'edc' on a qwerty keyboard, or 'erd' you will understand what I mean. The patheffort is a combination of two factors: hand alternation and column alternation. First, define a hand and column flag for a triad

    The definition of path effort here is arbitrary. I find that if the hands alternate between each keystroke, typing is easy (e.g. hf=0x). If both hands are used, but don't alternate then it's not as easy, particuarly when some of the columns in the triad are the same (e.g. same finger has to hit two keys like in "jeu"). If the same hand has to be used for three strokes then you're in trouble, particularly when some of the columns repeat. You can redefine the value of the path effort in <path_efforts> block.
    '''
    if not keyboard:
        raise ValueError('keyboard not defined')
    if not keytriads:
        raise ValueError('triads not defined')
    total_effort = 0
    contributing_triads = 0
    k = dict(k_params['effort'])
    for triad in keytriads:
        triad_effort = calculate_triad_effort(triad, keyboard, k, weight_params)
        num_triads = keytriads[triad]
        total_effort += triad_effort * num_triads
        contributing_triads += num_triads
        # if($triad =~ /[*]/) {
        # #printinfo("calculate_effort",$triad,$num_triads,$triad_effort,$totaleffort,$contributing_triads);
        # }
    total_effort /= contributing_triads
    #printinfo("calculate_effort_done",$totaleffort);
    #printinfo("*"x80);
    return total_effort

# %% calculate_triad_effort
# {

# my $effortlookup = {};

#   sub calculate_triad_effort {
#     my ($triad,$keyboard,$k1,$k2,$k3,$kb,$kp,$ks) = @_;

#     my $leaf = $keyboard->{map};
#     # characters of the triad
#     my ($c1,$c2,$c3)       = split(//,$triad);
#     my ($i1,$i2,$i3)       = ($leaf->{$c1}{idx},$leaf->{$c2}{idx},$leaf->{$c3}{idx});
#     if($CONF{memorize} && exists $effortlookup->{$i1}{$i2}{$i3}) {
#       return $effortlookup->{$i1}{$i2}{$i3};
#     } else {
#       #
#     }
#     # keyboard effort of each character
#     my ($be1,$be2,$be3)    = ($leaf->{$c1}{effort}{base},$leaf->{$c2}{effort}{base},$leaf->{$c3}{effort}{base});
#     my ($pe1,$pe2,$pe3)    = ($leaf->{$c1}{effort}{penalty},$leaf->{$c2}{effort}{penalty},$leaf->{$c3}{effort}{penalty});
#     # finger of each character
#     my ($f1,$f2,$f3)       = ($leaf->{$c1}{finger},$leaf->{$c2}{finger},$leaf->{$c3}{finger});
#     # row of each character
#     my ($row1,$row2,$row3) = ($leaf->{$c1}{row},$leaf->{$c2}{row},$leaf->{$c3}{row});
#     # hand of each character
#     my ($h1,$h2,$h3)       = ($leaf->{$c1}{hand},$leaf->{$c2}{hand},$leaf->{$c3}{hand});
#     # total triad effort is the sum of base effort product (finger distance) and penalty effort;
#     my $triad_effort       = $kb * $k1*$be1 * ( 1 + $k2*$be2 * ( 1 + $k3*$be3 ) ) + $kp * $k1*$pe1 * ( 1 + $k2*$pe2 * ( 1 + $k3*$pe3 ) );

#     if ($ks) {
#       # hand, finger, row flags for stroke path
#       # see http://mkweb.bcgsc.ca/carpalx/?typing_effort
#       my $hand_flag;
#       if($h1 == $h3) {
#   if($h2 == $h3) {
#     # same hand
#     $hand_flag = 2;
#   } else {
#     # alternating
#     $hand_flag = 1;
#   }
#       } else {
#   $hand_flag = 0;
#       }

#       my $finger_flag;

#       if( $f1 > $f2 ) {
#   if ( $f2 > $f3 ) {
#     # 1 > 2 > 3 - monotonic all different - pf=0
#     $finger_flag = 0;
#   } elsif ( $f2 == $f3 ) {
#     # 1 > 2 = 3 - monotonic some different - pf=1
#     if($c2 eq $c3) {
#       $finger_flag = 1;
#     } else {
#       $finger_flag = 6;
#     }
#   } elsif ( $f3 == $f1 ) {
#     $finger_flag = 4;
#   } elsif ( $f1 > $f3 && $f3 > $f2 ) {
#     # rolling
#     $finger_flag = 2;
#   } else {
#     # not monotonic all different - pf=3
#     $finger_flag = 3;
#   }
#       } elsif ( $f1 < $f2) {
#   if ( $f2 < $f3 ) {
#     # 1 < 2 < 3 - monotonic all different - pf=0
#     $finger_flag = 0;
#   } elsif ( $f2 == $f3 ) {
#     if($c2 eq $c3) {
#       # 1 < 2 = 3 - monotonic some different - pf=1
#       $finger_flag = 1;
#     } else {
#       $finger_flag = 6;
#     }
#   } elsif ( $f3 == $f1 ) {
#     # 1 = 3 < 2 - not monotonic some different - pf=2
#     $finger_flag = 4;
#   } elsif ($f1 < $f3 && $f3 < $f2) {
#     # rolling
#     $finger_flag = 2;
#   } else {
#     # not monotonic all different - pf=3
#     $finger_flag = 3;
#   }
#       } elsif( $f1 == $f2 ) {
#   if ( $f2 < $f3 || $f3 < $f1 ) {
#     # 1 = 2 < 3
#     # 3 < 1 = 2 - monotonic some different - pf=1
#     if($c1 eq $c2) {
#       $finger_flag = 1;
#     } else {
#       $finger_flag = 6;
#     }
#   } elsif ( $f2 == $f3 ) {
#     if($c1 ne $c2 && $c2 ne $c3 && $c1 ne $c3) {
#       $finger_flag = 7;
#     } else {
#       # 1 = 2 = 3 - all same - pf=4
#       $finger_flag = 5;
#     }
#   }
#       }

#     my $row_flag;

#       my @dr    = sort { ($b->[0] <=> $a->[0]) || ($a->[1] <=> $b->[1]) }
#   map { [abs($_),$_] } ($row1-$row2,$row1-$row3,$row2-$row3);
#       my ($drmax_abs,$drmax) = @{$dr[0]};
#       if ($row1 < $row2) {
#   if ($row3 == $row2) {
#     # 1 < 2 = 3 - downward with rep
#     $row_flag = 1;
#   } elsif ($row2 < $row3) {
#     # 1 < 2 < 3 - downward progression
#     $row_flag = 4;
#   } elsif ($drmax_abs == 1) {
#     $row_flag = 3;
#   } else {
#     # all/some different - delta row > 1
#     if($drmax < 0) {
#       $row_flag = 7;
#     } else {
#       $row_flag = 5;
#     }
#   }
#     } elsif ($row1 > $row2) {
#       if ($row3 == $row2) {
#   # 1 > 2 = 3 - upward with rep
#   $row_flag = 2;
#       } elsif ($row2 > $row3) {
#   # 1 > 2 > 3 - upward
#   $row_flag = 6;
#       } elsif ($drmax_abs == 1) {
#   $row_flag = 3;
#       } else {
#   if($drmax < 0) {
#     $row_flag = 7;
#   } else {
#     $row_flag = 5;
#   }
#       }
#     } else {
#       # 1=2
#       if($row2 > $row3) {
#   # 1 = 2 > 3 - upward with rep
#   $row_flag = 2;
#       } elsif ($row2 < $row3) {
#   # 1 = 2 < 3 - downward with rep
#   $row_flag = 1;
#       } else {
#   # all same
#   $row_flag = 0;
#       }
#     }

#     my $path_flag = "$hand_flag$row_flag$finger_flag";
#     my $path_cost = $CONF{effort_model}{weight_param}{penalties}{path_offset} + $CONF{effort_model}{path_cost}{$path_flag};
#     $triad_effort += $ks * $path_cost;

#     $CONF{debug} && printdebug(1,"triad $c1$c2$c3 keys $c1 $c2 $c3 base_effort $be1 $be2 $be3 penalty_effort $pe1 $pe2 $pe3 hand $h1 $h2 $h3 row $row1 $row2 $row3 finger $f1 $f2 $f3 ph $hand_flag pr $row_flag pf $finger_flag path $path_flag $path_cost effort $triad_effort");

#   } else {

#     $CONF{debug} && printdebug(1,"triad $c1$c2$c3 keys $c1 $c2 $c3 base_effort $be1 $be2 $be3 penalty_effort $pe1 $pe2 $pe3 row $row1 $row2 $row3 hand $h1 $h2 $h3 ph - pr - pf - path - 0 effort $triad_effort");

#   }
#   $effortlookup->{$i1}{$i2}{$i3} = $triad_effort;
#   return $triad_effort;

# }
# }

def calculate_triad_effort(triad, keyboard, k_params, weight_params=effort_weight_param):

    # TODO: find out how to store effortlookup
    # my $effortlookup = {};
    print(k_params)
    k1, k2, k3 = (float(k_params['k1']), float(k_params['k2']), float(k_params['k3']))
    kb, kp, ks = (float(k_params['kb']), float(k_params['kp']), float(k_params['ks']))

    leaf = keyboard['map']


    # characters of the triad
    try:
        c1, c2, c3 = [*triad]
    except ValueError as e:
        print(triad)
        raise e
    i1, i2, i3 = (leaf[c1]['idx'], leaf[c2]['idx'], leaf[c3]['idx'])

    # TODO: find where config.memorize can be defined
    # if($CONF{memorize} && exists $effortlookup->{$i1}{$i2}{$i3}) {
    #   return $effortlookup->{$i1}{$i2}{$i3};

    # keyboard effort of each character
    try:
        be1, be2, be3    = (leaf[c1]['effort']['base'], leaf[c2]['effort']['base'], leaf[c3]['effort']['base'])
        pe1, pe2, pe3    = (leaf[c1]['effort']['penalty'], leaf[c2]['effort']['penalty'], leaf[c3]['effort']['penalty'])
    except KeyError as e:
        print(c1, c2, c3)
        raise e

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
                row_flag = 3;
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
                    + float(effort_path_cost.get('path', path_flag).split(' #')[0])
        triad_effort += ks * path_cost

        print_debug(1, f'triad {triad}',
                    f'keys {c1} {c2} {c3}',
                    f'base_effort {be1} {be2} {be3}',
                    f'penalty_effort {pe1} {pe2} {pe3}',
                    f'hand {h1} {h2} {h3}',
                    f'row {row1} {row2} {row3}',
                    f'finger {f1} {f2} {f3}',
                    f'ph {hand_flag} pr {row_flag} pf {finger_flag}',
                    f'path {path_flag} {path_cost} effort {triad_effort}')

    else:
        print_debug(1, f'triad {triad}',
                    f'keys {c1} {c2} {c3}',
                    f'base_effort {be1} {be2} {be3}',
                    f'penalty_effort {pe1} {pe2} {pe3}',
                    f'row {row1} {row2} {row3}',
                    f'hand {h1} {h2} {h3}',
                    f'ph - pr - pf - path - 0 effort {triad_effort}')

    if i1 not in _effortlookup:
        _effortlookup.update({i1: {i2: {i3: triad_effort}}})
    elif i2 not in _effortlookup[i1]:
        _effortlookup[i1].update({i2: {i3: triad_effort}})
    elif i3 not in _effortlookup[i1][i2]:
        _effortlookup[i1][i2].update({i3: triad_effort})
    else:
        _effortlookup[i1][i2][i3] = triad_effort

    return triad_effort

# %% draw_keyboard

################################################################
#
# Create an image of the keyboard
#
################################################################

sub draw_keyboard {

  my $keyboard   = shift;
  my $file       = shift;
  my $parameters = shift;

  $file = resolve_path($file);

  my $imageparams = $CONF{imageparamsetdef}{ $CONF{imageparamset} };
  my $imagedetail = $CONF{imagedetaildef}{ $CONF{imagedetaillevel} };

  my $keysize    = $imageparams->{keysize};
  my $charxshift = $imageparams->{xshift};
  my $ucyshift   = $imageparams->{ucyshift};
  my $lcyshift   = $imageparams->{lcyshift};
  my $keymargin  = $imageparams->{keyspacing} < 1 ? $imageparams->{keyspacing} * $keysize : $imageparams->{keyspacing};
  my $shadow     = $imageparams->{shadowsize};

  my $width = (2+int(@{$keyboard->{key}[1]}))*($keysize+$keymargin);
  my $height = int(@{$keyboard->{key}})*($keysize+$keymargin)+$imageparams->{bottommargin};

  my $im = GD::Image->new($width,$height);

  my $colors = allocate_colors($im);

  $im->fill(0,0,$colors->{ $imageparams->{color}{background} });
  $im->rectangle(0,0,$width-1,$height-1,$colors->{ $imageparams->{color}{imageborder} } ) if $imagedetail->{imageborder};

  # get list of all unique costs

  my %costs;
  map {$costs{$_->{effort}{total}}++} map {@{$keyboard->{key}[$_]}} (0..@{$keyboard->{key}}-1);

  my @colors = split(/[\s,]+/,$imageparams->{color}{effort});

  # rank ordered list of costs
  my @costs_ranks = sort {$a <=> $b} keys %costs;

  my %costs_colors;
  my @keycolor_i = split(/[,\s]+/,$CONF{colors}{$imageparams->{color}{effort_color_i}});
  my @keycolor_f = split(/[,\s]+/,$CONF{colors}{$imageparams->{color}{effort_color_f}});
  for my $i (0..@costs_ranks-1) {
    my @rankcolor = map { scalar int(max($keycolor_i[$_] - $i/(@costs_ranks-1)*($keycolor_i[$_]-$keycolor_f[$_]))) } (0..2);
    my $colorname = "rankcolor$i";
    $colors->{$colorname} = $im->colorAllocate( @rankcolor );
    $costs_colors{$costs_ranks[$i]} = $colorname;
  }

  foreach my $row (0..@{$keyboard->{key}} - 1) {
    my $keyy = ($row+1)*$keymargin+$row*$keysize;
    foreach my $col (0..@{$keyboard->{key}[$row]}-1) {
      my @key_x0 = (0,
        1.5*$keysize+$keymargin,
        2*$keysize+$keymargin,
        2*($keysize+$keymargin)+0.25*$keysize,
       );
      my $key_x = (1+$col)*$keymargin + $keysize*($col) + $key_x0[$row];
      my $cost = $keyboard->{key}[$row][$col]->{effort}{total};
      my $hand = $keyboard->{key}[$row][$col]->{hand};
      my ($keycolour,$keybordercolour);
      if($imagedetail->{effortcolor}) {
  $keycolour = $costs_colors{$cost};
      } else {
  $keycolour = $imageparams->{color}{key};
      }
      $keybordercolour = $imageparams->{color}{keyborder};
      $im->filledRectangle($key_x+$shadow,$keyy+$shadow,
         $key_x+$keysize+$shadow,$keyy+$keysize+$shadow,
         $colors->{ $imageparams->{color}{keyshadow}} ) if $imagedetail->{keyshadow};
      $im->filledRectangle($key_x,$keyy,
         $key_x+$keysize,$keyy+$keysize,
         $colors->{$keycolour}) if $imagedetail->{fillkey};
      $im->rectangle($key_x,$keyy,
         $key_x+$keysize,$keyy+$keysize,
         $colors->{$keybordercolour}) if $imagedetail->{keyborder};
      # keys
    my $char_lc = $keyboard->{key}[$row][$col]->{lc};
    my $char_uc = $keyboard->{key}[$row][$col]->{uc};
    my $label_x = $key_x + $charxshift;
    my $labely = $keyy + $ucyshift;

    rendertext(image=>$im,
         font=>$CONF{font},
         x=>$label_x,
         y=>$labely,
         text=>$char_uc,
         size=>$imageparams->{fontsize},
         angle=>0,
         color=>$colors->{black}) if $imagedetail->{upcase} =~ /y/
     || ( $char_uc !~ /[A-Z]/ && $imagedetail->{upcase} eq "some");;

    rendertext(image=>$im,
         font=>$CONF{font},
         x=>$label_x,
         y=>$labely + $lcyshift,
         text=>$char_lc,
         size=>$imageparams->{fontsize},
         angle=>0,
         color=>$colors->{black}) if $imagedetail->{lowcase};

    rendertext(image=>$im,
         font=>$CONF{font},
         x=>$key_x+$keysize-16,
         y=>$keyy+$keysize-5,
         text=>sprintf("%.1f",$cost),
         size=>6,
         angle=>0,
         color=>$colors->{black}) if $imagedetail->{effort};

    rendertext(image=>$im,
         font=>$CONF{font},
         x=>$key_x+$keysize-7,
         y=>$keyy+$keysize-15,
         text=> $hand?"R":"L",
         size=>6,
         angle=>0,
         color=>$colors->{black}) if $imagedetail->{hand};

    rendertext(image=>$im,
         font=>$CONF{font},
         x=>$key_x+$keysize-7,
         y=>$keyy+$keysize-25,
         text=> $keyboard->{key}[$row][$col]{finger},
         size=>6,
         angle=>0,
         color=>$colors->{black}) if $imagedetail->{finger};

    sub rendertext {
  my %args = @_;
  $args{text} = uc $args{text} if $imagedetail->{capitalize};
  #printinfo($args{font});
  $args{image}->stringFT(@args{qw(color font size angle x y text)});
    }

  }
}

  if($parameters && $imagedetail->{parameters}) {
    my @text;
    for my $parameter (sort keys %$parameters) {
      my $format;
      my $value = $parameters->{$parameter};
      if($value =~ /-?\d+\.\d+/) {
  $format = "%.5f";
      } else {
  $format = "%s";
      }
      push @text, sprintf("%s = $format",$parameter,$value);
    }
    rendertext(image=>$im,
         font=>$CONF{fontc},
         x=>5,
         y=>$height - 10,
         text=> join(" :: ", @text),
         size=>10,
         angle=>0,
         color=>$colors->{black});
  }

  printdebug(1,"creating keyboard image",$file);
  open(IM,">$file") || die "cannot open png file $file for writing";
  binmode IM;
  print IM $im->png;
  close(IM);
}

# %% allocate_colors

sub allocate_colors {
  my $image = shift;
  my $colors;
  foreach my $color (keys %{$CONF{colors}}) {
    my $colorvalue = $CONF{colors}{$color};
    $colors->{$color} = $image->colorAllocate(split(/[, ]+/,$colorvalue));
  }
  return $colors;
}

# %% read_document
# =pod

# =head2 read_document()

#  $triads = read_document(); # create hashref to triad frequencies

#  $triads->{aab} = frequency of aab;
#  $triads->{abc} = frequency of abc;

# Read a document from file and create a list of of character triads. Triads are overlapping (more on overlapping below) 3-character combinations. Each triad is stored along with the number of times it appears in the document. All triads are stored, including overlapping triads.

# For example, if the document line is

#   I am a very lazy dog with big ears.

# Then the triads will be

#   i am              iam
#     am a            ama
#      m a v          mav
#        a ve         ave
#          ver        ver
#           ery       ery
#            ry l     ryl
#             y la    yla
#   ...

# and so on. Notice that spaces in the document are disregarded during construction of the triads.

# Depending on the parse mode, the input document undergoes some transformation before triads are constructed. Each mode must be defined using a <mode_def> block. Three modes are defined and you can add more.

# You can control how the triads are read by the <triad> block

#   <triad>
#   maxnum = 1000 # limit number of triads
#   overlap = yes # if set to yes, a triad potentially begins at each character (triads overlap by maximum of 2 characters)
#                 # if set to no, triads abut
#   </triad>

# =over

# =item mode = perl

# If the mode is set to "perl", then all comment lines are disregarded. Comments are identified by lines that begin with #.

# =item mode = english

# English mode removes all non-alphanumeric characters before constructing triads.

# =item mode = letter

# All non-letter characters are removed and remaining letters are switched to lower case.

# =back

# =cut

# sub read_document {

#   # prepend script's path before relative paths to the input text
#   my $file = shift;
#   my $args = shift || {};
#   $file = resolve_path($file);

#   die "cannot find input document [$file]" unless -e $file;
#   die "no permission to read input document [$file]" unless -r $file;

#   my $keytriads;
#   my $triads_count=0;

#   ################################################################
#   # read from corpus caches, if available
#   my $file_cache       = "$file.cache";
#   my $file_cache_chars = "$file.chars.cache";
#   if(-e $file_cache && ! $args->{charlist}) {
#     printdebug(1,"reading triads from cache",$file_cache);
#     return retrieve($file_cache);
#   } elsif (-e $file_cache_chars && $args->{charlist}) {
#     printdebug(1,"reading character list from cache",$file_cache_chars);
#     return retrieve($file_cache_chars);
#   }

#   open(INPUT,$file) || die "cannot open input document [$file]";

#   my $charlist;

#  READLINE: while(my $line = <INPUT>) {
#     chomp $line;
#     # remove spaces from text - assume that a space does not influence effort of typing
#     printdebug(2,"read_document","pre-processed", $line);
#     $line =~ s/\s//g;
#     next unless $line;
#     if($CONF{mode_def}{$CONF{mode}}{force_case} eq "lc") {
#       $line =~ tr/A-Z/a-z/;
#     } elsif($CONF{mode_def}{$CONF{mode}}{force_case} eq "uc") {
#       $line =~ tr/a-z/A-Z/;
#     }
#     if($CONF{mode_def}{$CONF{mode}}{reject_char_rx}) {
#       (my $rx = $CONF{mode_def}{$CONF{mode}}{reject_char_rx}) =~ s/\/\//\//g;
#       $line =~ s/$rx//g;
#     }
#     if($CONF{mode_def}{$CONF{mode}}{reject_line_rx}) {
#       (my $rx = $CONF{mode_def}{$CONF{mode}}{reject_line_rx}) =~ s/\/\//\//g;
#       if($line =~ /$rx/) {
#         printdebug(2,"read_document","skipping line",$line);
#         next READLINE;
#       }
#     }
#     if($CONF{mode_def}{$CONF{mode}}{accept_line_rx}) {
#       (my $rx = $CONF{mode_def}{$CONF{mode}}{accept_line_rx}) =~ s/\/\//\//g;
#       if($line !~ /$rx/) {
#         printdebug(2,"read_document","skipping line",$line);
#         next READLINE;
#       }
#     }
#     printdebug(2,"read_document","processed", $line);

#     push @$charlist, split("",$line);

#     if($args->{charlist} && $CONF{charlist_max_length} && @$charlist > $CONF{charlist_max_length}) {
#       return $charlist;
#     }

#     while($line =~ /(...)/g) {
#       my $triad = $1;
#       if(substr($triad,0,1) eq substr($triad,1,1) &&
#          substr($triad,1,1) eq substr($triad,2,1) &&
#          ! $CONF{mode_def}{$CONF{mode}}{accept_repeats}) {
#         # identical triplet - skip if requested
#         printdebug(3,"skipping repeated triad $triad");
#       } else {
#         $keytriads->{$triad}++;
#         $triads_count++;
#         printdebug(3,"accepting $triad");
#         if($CONF{triads_max_num} && $triads_count >= $CONF{triads_max_num}) {
#           $CONF{debug} && printdebug(1,"limiting triads to $CONF{triads_max_num}");
#           last READLINE;
#         }
#       }
#       pos $line -= 2 if $CONF{triads_overlap};
#     }
#   }
#   printdebug(1,"found $triads_count triads (",int(keys %$keytriads),"unique)");
#   # remove low-frequency triads
#   if($CONF{triads_min_freq}) {
#     for my $triad (keys %$keytriads) {
#       if($keytriads->{$triad} < $CONF{triads_min_freq}) {
#         $CONF{debug} && printdebug(1,"removing rare triad",$triad,"freq",$keytriads->{$triad});
#         delete $keytriads->{$triad};
#       }
#     }
#   }
#   # limit number of triads
#   if($CONF{triads_max_num}) {
#     $CONF{debug} && printdebug(1,"limiting triads to $CONF{triads_max_num}");
#     my @triads_by_freq = sort {$keytriads->{$b} <=> $keytriads->{$a}} keys %$keytriads;
#     for my $i ($CONF{triads_max_num} .. @triads_by_freq-1) {
#       my $triad_to_del = $triads_by_freq[$i];
#       $CONF{debug} && printdebug(1,"removing triad",$triad_to_del,"freq",$keytriads->{$triad_to_del});
#       delete $keytriads->{$triad_to_del};
#     }
#   }
#   printdebug(1,"writing triads to cache",$file_cache);
#   store($keytriads,$file_cache);

#   # process the character list to create a frequency table of
#   # characters and character transitions
#   store($charlist,$file_cache_chars);
#   return $keytriads;
# }

def read_document(file, charlist=False):
    '''Read a document from file and create a list of of character triads. Triads are overlapping (more on overlapping below) 3-character combinations. Each triad is stored along with the number of times it appears in the document. All triads are stored, including overlapping triads.

    triads = read_document(xyz) # create dict of triad frequencies

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

    Depending on the parse mode, the input document undergoes some transformation before triads are constructed. Each mode must be defined using a <mode_def> block. Three modes are defined and you can add more.

    You can control how the triads are read by the <triad> block

    <triad>
    maxnum = 1000 # limit number of triads
    overlap = yes # if set to yes, a triad potentially begins at each character (triads overlap by maximum of 2 characters)
                    # if set to no, triads abut
    </triad>

    =over

    =item mode = perl

    If the mode is set to "perl", then all comment lines are disregarded. Comments are identified by lines that begin with #.

    =item mode = english

    English mode removes all non-alphanumeric characters before constructing triads.

    =item mode = letter

    All non-letter characters are removed and remaining letters are switched to lower case.'''

    # prepend script's path before relative paths to the input text
    file = resolve_path(file)
    if not os.path.exists(file):
        raise FileNotFoundError(f'cannot find input document {file}')

    # die "no permission to read input document [$file]" unless -r $file;

    keytriads = defaultdict(int)
    triads_count = 0

    ################################################################
    # read from corpus caches, if available
    file_cache = file + '.cache'
    file_cache_chars = file + '.chars.cache'

    if os.path.exists(file_cache) and not get_charlist:
        print_debug(1, f'reading triads from cache {file_cache}')
        with open(file_cache, 'r') as f:
            cache = json.load(f)
        return cache
    elif os.path.exists(file_cache_chars) and not get_charlist:
        print_debug(1, f'reading character list from cache {file_cache_chars}')
        with open(file_cache_chars, 'r') as f:
            cache = json.load(f)
        return cache

    with open(file, 'r') as f:
        input_file = f.read().splitlines()
    charlist = []

    for idx, line in enumerate(input_file):
        # remove spaces from text - assume that a space does not influence effort of typing
        line = ''.join(line.split())
        if not line.strip():
            continue
        print_debug(2, 'read_document', 'pre-processed', line)

        mode = f'mode_def {config.get("corpus", "mode")}'
        mode_cfg = kb_modes[mode]

        if mode_cfg['force_case'] == 'lc':
            line = line.lower()
        elif mode_cfg['force_case'] == 'uc':
            line = line.lower()

        if kb_modes.has_option(mode, 'reject_char_rx'):
            rx = re.compile(mode_cfg['reject_char_rx'])
            line = re.sub(rx, '', line)
        if kb_modes.has_option(mode, 'reject_line_rx'):
            rx = re.compile(mode_cfg['reject_line_rx'])
            line = re.sub(rx, '', line)
            if not line:
                print_debug(2, 'read_document', 'skipping line', line)
                continue
        if kb_modes.has_option(mode, 'accept_line_rx'):
            rx = re.compile(mode_cfg['accept_line_rx'])
            if not re.findall(rx, line):
                print_debug(2, 'read_document', 'skipping line', line)
                continue
        print_debug(2, 'read_document', 'processed line', line)

        charlist.append([*line])

        # TODO: find where charlist_max_length might be defined
        if get_charlist:
            if not config.has_options('options', 'charlist_max_length'):
                return charlist
            elif len(charlist) >= config.getint('options', 'charlist_max_length'):
                return charlist
            else:
                pass

        line_triads = [line[idx:idx+3] for idx, item in enumerate(line)]
        line_triads = [triad for triad in line_triads if len(triad) == 3]
        iter_triads = iter(line_triads)
        for triad in iter_triads:
            if len(set(triad)) == 1 and not kb_modes.getboolean(mode, 'accept_repeats'):
                # identical triplet - skip if requested
                print_debug(3, f'skipping repeated triad {triad}')
                line_triads.remove(triad)
                continue
            keytriads[triad] += 1
            triads_count += 1
            print_debug(3, f'accepting {triad}')
            # suppressed, treating below when ordering by frequency
            # if config.has_option('corpus', 'triads_max_num') and triads_count >= config.get('corpus', 'triads_max_num'):
            #     print_debug(1, f'limiting triads to {config.get("corpus", "triads_max_num")}')
            #     break
            # TODO: triads_overlap might be in config.corpus or config.options
            # also, review this modus operandi
            if config.getboolean('corpus', 'triads_overlap'):
                next(islice(iter_triads, 2, 2), None)

    print_debug(1, f'found {triads_count} triads ({len(keytriads)} unique)')
    # remove low-frequency triads
    min_freq = 0
    if config.has_option('corpus', 'triads_min_freq'):
        min_freq = config.getint('corpus', 'triads_min_freq')
    for triad, freq in keytriads.copy().items():
        if freq < min_freq:
            print_debug(1, f'removing rare triad {triad} freq {freq}')
            del keytriads(triad)

    # limit number of triads
    # TODO: triads_max_num might be in config.corpus or config.options
    if config.has_option('corpus', 'triads_max_num'):
        print_debug(1, f'limiting triads to {config.get("corpus", "triads_max_num")}')
        triads_by_freq = dict(sorted(keytriads.items(), key=lambda item: item[1], reverse=True))
        for triad_to_del in list(triads_by_freq.keys())[config.getint('corpus', 'triads_max_num'):]:
            print_debug(1, f'removing triad {triad_to_del} freq {keytriads[triad_to_del]}')
            del keytriads[triad_to_del]

    print_debug(1, 'writing triads to cache', file_cache)
    with open(file_cache, 'w') as cache_file:
        cache_file.write(json.dumps(keytriads))

    # process the character list to create a frequency table of
    # characters and character transitions
    with open(file_cache_chars, 'w') as cache_chars:
        cache_chars.write(json.dumps(charlist))

    return keytriads

# %% printkeyboard

################################################################
#
# Dump the keyboard layout to STDOUT
#
################################################################

sub printkeyboard {
  my $keyboard = shift;
  return if $CONF{stdout_quiet};
  print "-"x60,"\n";
  foreach my $row (0..@{$keyboard->{key}}-1) {
    my $lcrow = join(" ",map {$_->{lc}} @{$keyboard->{key}[$row]});
    my $ucrow = join(" ",map {$_->{uc}} @{$keyboard->{key}[$row]});
    print $lcrow," "x(30-length($lcrow)),$ucrow,"\n";
  }
  print "-"x60,"\n";
}
# %% _parse_finger_distance
################################################################
#
# Load effort of hitting each key.
#
# The values here represent the baseline effort. You can add
# a effort offset in create_keyboard(), effectively adding
# a constant to all effort values.
#
# Home row keys require the least effort and therefore have
# the lowest effort. Keys assigned to the pinky have a high
# effort, especially if hitting them also requires wrist
# rotation (e.g. z requires wrist rotation but [ does not).
#
################################################################

# sub _parse_finger_distance {
#   my $cost;
#   my $row = 0;
#   my $leaf = $CONF{effort_model}{finger_distance}{row};
#   die "typing effort for keyboard rows not defined - you need <effort_row ROW> blocks" unless ref($leaf) eq "HASH";
#   for my $row_idx (keys %$leaf) {
#     my @keycost = split(" ",$leaf->{$row_idx}{effort});
#     map { $cost->[$row_idx-1][ $_ ] = $keycost[$_] } (0..@keycost-1);
#   }
#   return $cost;
# }

def _parse_finger_distance():
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
    leaf = dict(effort_finger_distance['effort'])
    # die "typing effort for keyboard rows not defined - you need <effort_row ROW> blocks" unless ref($leaf) eq "HASH";
    for row in leaf.keys():
        keycost = leaf[row].split()
        cost.update({int(row.split()[-1])-1: [float(cost) for cost in keycost]})

    return cost


# %% _parse_keyboard_layout

# =pod

# =head2 _parse_keyboard_layout

# Parse the <keyboard><row> blocks to determine the location of keys on the keyboard.

# Keyboard structure is stored as a hashref. Each key is stored by row/col position

#   $keyboard->{key}[ROW][COL]{lc}     = lower case at ROW,COL
#   $keyboard->{key}[ROW][COL]{uc}     = upper case at ROW,COL
#   $keyboard->{key}[ROW][COL]{finger} = finger for hitting key at ROW,COL

#   $keyboard->{map}{CHAR}{row}    = ROW for key CHAR
#   $keyboard->{map}{CHAR}{col}    = COL for key CHAR
#   $keyboard->{map}{CHAR}{case}   = CASE of key CHAR
#   $keyboard->{map}{CHAR}{finger} = FINGER for hitting key CHAR

# =cut

# sub _parse_keyboard_layout {
#   my $keyboardfile = shift;
#   $keyboardfile = resolve_path($keyboardfile);
#   die "cannot file keyboard definition - $keyboardfile" unless -e $keyboardfile;
#   my %keyboard = _parse_conf_file($keyboardfile);
#   die "no keyboard row definitions in keyboard layout" unless $keyboard{keyboard}{row};
#   my $keyboard;
#   for my $row_idx (sort {$a <=> $b} keys %{$keyboard{keyboard}{row}}) {
#       die "no keys defined in keyboard layout for row $row_idx" unless $keyboard{keyboard}{row}{$row_idx}{keys};
#       die "no fingers defined in keyboard layout for row $row_idx" unless $keyboard{keyboard}{row}{$row_idx}{fingers};
#       my @keys    = split(" ",$keyboard{keyboard}{row}{$row_idx}{keys});
#       my @fingers = split(" ",$keyboard{keyboard}{row}{$row_idx}{fingers});
#       my $row = $row_idx - 1;
#       my $col = 0;
#       for my $key_idx (0..@keys-1) {
#     my $key = $keys[$key_idx];
#     my $finger = $fingers[$key_idx];
#     die "undefined finger assignment for keyboard layout for row,col $row_idx,$col" unless defined $finger;
#     my $hand   = $finger > 5 ? 1 : 0;
#     my @char = split(//,$key);
#     # letter keys have lowercase/uppercase characters on same key
#     if($char[0] =~ /[a-z]/i) {
#         $char[1] = uc $char[0];
#     } else {
#         if(@char != 2) {
#       die "keyboard layout broken - non-letter key $key must have have two characters";
#         }
#     }
#     printdebug(2,"_parse_keyboard_layout","keyassignment","lc",$char[0],"uc",$char[1],"to row,col",$row,$col);

#     @{$keyboard->{key}[$row][$col]}{qw(row col lc uc finger hand)} = ($row,$col,@char,$finger,$hand);

#     for my $case (0,1) {
#         @{$keyboard->{map}{$char[$case]}}{qw(row col case finger hand)} = ($row,$col,$case,$finger,$hand);
#     }

#     $col++;
#       }
#   }
#   return $keyboard;
# }

def _parse_keyboard_layout(keyboard_file):
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

    keyboard_file = resolve_path(keyboard_file)
    if not os.path.exists(keyboard_file):
        raise FileNotFoundError(f'cannot find keyboard definition file {keyboard_file}')

    keyboard = parse_conf_file(keyboard_file)
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
                print_debug(2, '_parse_keyboard_layout', 'keyassignment', 'lc', key[0], 'uc', key[1], 'to row,col', row, col)
                keyboard_def['key'][row].update({col: {'lc': key[0], 'uc': key[1], 'finger': finger, 'hand': hand}})

                for idx, case in enumerate(['lc', 'uc']):
                    keyboard_def['map'].update({key[idx]: {'row': row, 'col': col, 'case': case, 'finger': finger, 'hand': hand}})

    return keyboard_def


# %% _parse_mask

################################################################
#
# Parse the keyboard mask, defined in <mask_row N> blocks in the
# configuration file. Each row block should contain an entry
#
# mask = M M M M M ...
#
# where M = 1|0 depending on whether you want the key to be
# eligible (1) or not eligible (0) for remapping. There should
# be as many Ms in each row as there are columns on the keyboard
#
################################################################

sub _parse_mask {
  my $mask;
  for my $row (sort {$a <=> $b} keys %{$CONF{mask_row}}) {
    my $row_idx = $row - 1;
    my @row_mask = split(/[\s]+/,$CONF{mask_row}{$row}{mask});
    for my $col_idx (0..@row_mask-1) {
      $mask->[$row_idx][$col_idx] = $row_mask[$col_idx];
    }
  }
  return $mask;
}

# %% make_relocatable_list

=pod

=head2 make_relocatable_list()

  my $list = make_relocatable_list($mask)

Based on the key mask generated by _parse_mask(), this function returns a list of all keys that can be relocated. The list is a set of row,col pairs.

  $list = [ ... [row,col], [row,col], ... ]

=cut

sub make_relocatable_list {
  my $mask = shift;
  my $list;
  foreach my $row (0..@$mask-1) {
    foreach my $col (0..@{$mask->[$row]}-1) {
      if($mask->[$row][$col]) {
  push @$list, [$row,$col];
      }
    }
  }
  return $list;
}

# %% create_keyboard

# =pod

# =head2 create_keyboard()

#   my $keyboard = create_keyboard();

# Parses the keyboard layout and creates an array that keeps track of the keys, their positions, character assignments and typing effort. The keyboard array is indexed by row and column of the key and contains a hash

#   $keyboard->{key}[row][col]
#                            {lc}
#                            {uc}
#                            {row}
#                            {col}
#                            {effort}

# The keyboard layout is read from the <keyboard><row> blocks. The effort in the {key} part of the keyboard object is the canonical effort for the row,col combination as defined in <effort_row> plus any baseline and hand penalties.

# The keymap hash is a direct mapping between a character and its position and hand assignment on the keyboard

#   $keyboard->{map}{CHAR}
#                        {row}
#                        {col}
#                        {hand}
#                        {effort}

# The effort in the {map} part of the keyboard object is the effort for the character, based on its row,col combination and includes the shift penalty.

# For the standard qwerty layout, look at the keyboard you're using right now (true for >99% of typists). For Dvorak layout, see http://www.mwbrooks.com/dvorak.

# =cut

# sub create_keyboard {

#   my $keyboard_type = shift;
#   my $keyboard      = _parse_keyboard_layout($keyboard_type);
#   my $cost          = _parse_finger_distance();

#   # merge the cost into the keyboard hash

#   my $keyidx=0;
#   for my $row_idx (0..@{$keyboard->{key}}-1) {
#     for my $col_idx (0..@{$keyboard->{key}[$row_idx]}-1) {

#       # this is the canonical cost of typing a key defined in the <effort_row> blocks

#       my $base_effort = $cost->[$row_idx][$col_idx];
#       my $finger      = $keyboard->{key}[$row_idx][$col_idx]{finger};
#       my $hand        = $keyboard->{key}[$row_idx][$col_idx]{hand};
#       my $row         = $row_idx;

#       $keyboard->{key}[$row_idx][$col_idx]{idx} = $keyidx;

#       die "create_keyboard - there is a key defined at row,col $row_idx,$col_idx but no associated effort in the effort file" unless defined $base_effort;
#       die "create_keyboard - there is a key defined at row,col $row_idx,$col_idx but no finger assignment" unless defined $finger;
#       die "create_keyboard - there is a key defined at row,col $row_idx,$col_idx but no hand assignment" unless defined $hand;

#       my $hand_penalty   = $CONF{effort_model}{weight_param}{penalties}{hand}{ $hand ? "right" : "left" };
#       my $finger_penalty = $finger < 5 ?
#   (split(/[,\s]+/,$CONF{effort_model}{weight_param}{penalties}{finger}{left}))[$finger] :
#     (split(/[,\s]+/,$CONF{effort_model}{weight_param}{penalties}{finger}{right}))[$finger-5];
#       my $row_penalty    = $CONF{effort_model}{weight_param}{penalties}{row}{$row};

#       my ($penalty_effort,$total_effort);

#       my $penalty_effort  =
#   $CONF{effort_model}{weight_param}{penalties}{default} +
#     $CONF{effort_model}{weight_param}{penalties}{weight}{hand}*$hand_penalty +
#       $CONF{effort_model}{weight_param}{penalties}{weight}{row}*$row_penalty +
#         $CONF{effort_model}{weight_param}{penalties}{weight}{finger}*$finger_penalty;

#       $total_effort = $CONF{effort_model}{k_param}{kb} * $base_effort + $CONF{effort_model}{k_param}{kp} * $penalty_effort;

#       for my $case (0,1) {
#   $CONF{debug} && printdebug(1,
#            "create_keyboard","effortassign",
#            "key",$keyboard->{key}[$row_idx][$col_idx]{ $case ? "uc" : "lc"},
#            "at row,col",$row_idx,$col_idx,
#            "hand",$hand,
#            "row",$row,
#            "finger",$finger,
#            "base_effort",$base_effort,
#            "base_penalty",$CONF{effort_model}{weight_param}{penalties}{default},
#            "hand_penalty",$CONF{effort_model}{weight_param}{penalties}{weight}{hand},$hand_penalty,
#            "row_penalty",$CONF{effort_model}{weight_param}{penalties}{weight}{row},$row_penalty,
#            "finger_penalty",$CONF{effort_model}{weight_param}{penalties}{weight}{finger},$finger_penalty,
#            "total_penalty",$penalty_effort,
#            "total_effort",$total_effort);
#       }

#       $keyboard->{key}[$row_idx][$col_idx]{effort}{total}   = $total_effort;
#       $keyboard->{key}[$row_idx][$col_idx]{effort}{base}    = $base_effort;
#       $keyboard->{key}[$row_idx][$col_idx]{effort}{penalty} = $penalty_effort;

#       $keyboard->{map}{ $keyboard->{key}[$row_idx][$col_idx]{"uc"} }{effort}{total} = $total_effort;
#       $keyboard->{map}{ $keyboard->{key}[$row_idx][$col_idx]{"lc"} }{effort}{base} = $base_effort;
#       $keyboard->{map}{ $keyboard->{key}[$row_idx][$col_idx]{"lc"} }{effort}{penalty} = $penalty_effort;
#       $keyboard->{map}{ $keyboard->{key}[$row_idx][$col_idx]{"uc"} }{idx}    = $keyidx;
#       $keyboard->{map}{ $keyboard->{key}[$row_idx][$col_idx]{"lc"} }{idx}    = $keyidx;

#       $keyidx++;
#     }
#   }
#   return $keyboard;
# }

def create_keyboard(keyboard_type):
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

    keyboard = _parse_keyboard_layout(keyboard_type)
    cost = _parse_finger_distance()

    # merge the cost into the keyboard hash

    base_penalty = effort_weight_param.getfloat('main', 'default')
    hand_weight = effort_weight_param.getfloat('weight', 'hand')
    row_weight = effort_weight_param.getfloat('weight', 'row')
    finger_weight = effort_weight_param.getfloat('weight', 'finger')

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
            hand_penalty = effort_weight_param.getfloat('hand', hand_descr)
            finger_penalty = float(effort_weight_param.get('finger', hand_descr).split()[finger if hand == 0 else finger-5])
            row_penalty = effort_weight_param.getfloat('row', str(row))

            penalty_effort = base_penalty \
                             + hand_weight * hand_penalty \
                             + row_weight * row_penalty \
                             + finger_weight * finger_penalty

            total_effort = effort_k_param.getfloat('effort', 'kb') * base_effort + effort_k_param.getfloat('effort', 'kp') * penalty_effort

            for case in ['lc', 'uc']:
                print_debug(1,
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

# %% populate_configuration

# sub populateconfiguration {
#   foreach my $key (keys %OPT) {
#     $CONF{$key} = $OPT{$key};
#   }

#   # any configuration fields of the form __XXX__ are parsed and replaced with eval(XXX). The configuration
#   # can therefore depend on itself.
#   #
#   # flag = 10
#   # note = __2*$CONF{flag}__ # would become 2*10 = 20

#   _parse_conf_data(\%CONF);
# }

def populate_configuration():
    for key in vars(options):
        if getattr(options, key) is not None:
            if not 'options' in config:
                config['options'] = {}
            config['options'][key] = str(getattr(options, key))

    # any configuration fields of the form __XXX__ are parsed and replaced with eval(XXX). The configuration
    # can therefore depend on itself.
    #
    # flag = 10
    # note = __2*$CONF{flag}__ # would become 2*10 = 20

    _parse_conf_data()

# %% _parse_conf_data

# sub _parse_conf_data {
#   my $conf = shift;
#   for my $key (keys %$conf) {
#     my $value = $conf->{$key};
#     if(ref($value) eq "HASH") {
#       _parse_conf_data($value);
#     } else {
#       while($value =~ /__([^_].+?)__/g) {
#         my $source = "__" . $1 . "__";
#         my $target = eval $1;
#         $value =~ s/\Q$source\E/$target/g;
#       }
#       if($value =~ /^eval\((.*)\)$/) {
#         $conf->{$key} = eval $1;
#       } else {
#         $conf->{$key} = $value;
#       }
#     }
#   }
# }

def _parse_conf_data():
    for section in config.sections():
        for key, value in config[section].items():
            pattern = re.compile(r'__([^_].+?)__')
            cmd = pattern.search(value)
            if cmd is not None:
                cmd = eval(cmd.group(1))
                config[section][key] = re.sub(pattern, cmd, value)
            else:
                pass

# %% parse_conf_file

# sub _parse_conf_file {
#   my $file = shift;
#   if(-e $file && -r _) {
#     my $conf_obj = new Config::General(-ConfigFile=>$file,
# 				       -AllowMultiOptions=>1,
# 				       -LowerCaseNames=>1,
# 				       -IncludeRelative=>1,
# 				       -ConfigPath=>[ "$FindBin::RealBin/../etc" ],
# 				       -AutoTrue=>1);
#     return $conf_obj->getall;
#   } else {
#     die "cannot find or read configuration file $file";
#   }
# }

def parse_conf_file(file, filepath='./etc'):
    if os.path.isfile(file):
        file = file
    else:
        file = os.path.join(filepath, file)
    if not os.path.exists(file):
        raise FileNotFoundError(f'cannot find or read configuration file {file}')

    conf_file = configparser.ConfigParser()
    conf_file.read(file)

    # TODO: used for debugging, remove later
    for section in conf_file.sections():
        for option in conf_file[section]:
            if conf_file.get(section, option).endswith('.conf'):
                nested_file = conf_file.get(section, option)
                print(f'make sure to import {nested_file} related to {file}')

    return conf_file

# %% load_configuration

# sub loadconfiguration {
#   my $file = shift;
#   my ($scriptname) = fileparse($0);
#   $file ||= "$scriptname.conf";
#   my @confpath = ("$FindBin::RealBin/../etc/",
# 		  "$FindBin::RealBin/etc/",
# 		  "$FindBin::RealBin/",
# 		  "$ENV{HOME}/.carpalx/etc",
# 		  "$ENV{HOME}/.carpalx");
#   if(-e $file && -r _) {
#     # great the file exists
#   } else {
#     my $found;
#     for my $path (@confpath) {
#       my $tryfile = "$path/$file";
#       if (-e $tryfile && -r _) {
# 	$file = $tryfile;
# 	$found=1;
# 	last;
#       }
#     }
#     return undef unless $found;
#   }
#   my $conf = new Config::General(-ConfigFile=>$file,
# 				 -AllowMultiOptions=>"yes",
# 				 -IncludeRelative=>"yes",
# 				 -LowerCaseNames=>1,
# 				 -ConfigPath=>\@confpath,
# 				 -AutoTrue=>1);
#   %CONF = $conf->getall;
#   $OPT{configfile} = $file;
#   ($CONF{configfile}) = grep($_ =~ /$file$/, $conf->files());
#   if($CONF{configfile} !~ /^\//) {

#     $CONF{configfile} = sprintf("%s/%s",getcwd,$CONF{configfile});
#   }
#   $CONF{configdir} = dirname($CONF{configfile});
# }

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

    config = parse_conf_file(filepath)
    config['main']['configfile'] = filename
    config['main']['configdir'] = os.path.abspath(dir)

    return config

# %% print debug

################################################################
#
# $CONF{debug} && printdebug(1,"this is level 1 debug");
# $CONF{debug} && printdebug(2,"this is level 2 debug");
# ...
#
################################################################

# sub printdebug {
#   my $level = shift;
#   if($CONF{debug} >= $level) {
#     printinfo("debug",@_);
#   }
# }

def print_debug(level, *messages):
    messages = [str(msg) for msg in messages]
    if int(config.get('options', 'debug')) >= level:
        print(f'DEBUG: {" ".join(messages)}')

# suppressed for now
# sub printinfo {
#   print join(" ",@_),"\n";
# }

# %% main

def main():

    parser = argparse.ArgumentParser()
    opt = parser.add_argument_group('options')
    opt.add_argument('-keyboard_input', action='store')
    opt.add_argument('-keyboard_output', action='store')
    opt.add_argument('-action', action='store')
    opt.add_argument('-corpus', action='store')
    opt.add_argument('-words', action='store')
    opt.add_argument('-wordlength', action='store')
    opt.add_argument('-detail', action='store_true')
    opt.add_argument('-mode', action='store')
    opt.add_argument('-triads_max_num', action='store')
    opt.add_argument('-triads_overlap', action='store_true')
    opt.add_argument('-cdump', action='store_true')
    opt.add_argument('-configfile', action='store')
    opt.add_argument('-man', action='store_true')
    opt.add_argument('-debug', type=int, action='store', default=1)

    if False:
        options = parser.parse_args()
    else: # TODO: only for debugging
        os.chdir(r'C:\Users\cleisonp\OneDrive - Alcast do Brasil SA\Documentos\GitHub\carpalx-py')
        options = parser.parse_args(r'-configfile etc/tutorial-00.ini'.split())

    # pod2usage() if $OPT{help};
    # if options.man:
    #   pod2usage(-verbose=>2) if $OPT{man};
    config = load_configuration(options.configfile)
    options.configfile = config.get('main', 'configfile')

    # TODO: only for debugging
    config['options']['debug'] = '2'

    keyboard_input = parse_conf_file(config.get('kb_definition', 'keyboard_input'))
    effort_model = parse_conf_file(config.get('model', 'effort_model'))
    effort_k_param = parse_conf_file(effort_model.get('main', 'k_param'))
    effort_weight_param = parse_conf_file(effort_model.get('main', 'weight_param'))
    effort_path_cost = parse_conf_file(effort_model.get('main', 'path_cost'))
    effort_finger_distance = parse_conf_file(effort_model.get('main', 'finger_distance'))

    colors = parse_conf_file(config.get('kb_parameters', 'colors'))
    kb_mask = parse_conf_file(config.get('kb_parameters', 'mask'))
    kb_modes = parse_conf_file(config.get('kb_parameters', 'modes'))

    populate_configuration() # copy command line options to config hash
    # validateconfiguration();
    # if($CONF{cdump}) {
    #   $Data::Dumper::Pad = "debug parameters";
    #   $Data::Dumper::Indent = 1;
    #   $Data::Dumper::Quotekeys = 0;
    #   $Data::Dumper::Terse = 1;
    #   print Dumper(\%CONF);
    #   exit;
    # }

    actions = config.get('main', 'action').split(',')
    _effortlookup = {}

    # my ($keytriads,$keyboard);

    for action in actions:
        print_debug(1, f'found action {action}')
        match action:
            case 'loadtriads':
                ################################################################
                #
                # read the document and extract triads
                #
                # triads are adjacent three-key combinations parsed from the
                # document based on the setting of the mode=MODE value
                # (see <mode MODE> block for filters for the MODE).
                #
                print_debug(1, 'loading triad from corpus file', config.get('corpus', 'corpus'))
                keytriads = read_document(config.get('corpus', 'corpus'))
            case 'loadkeyboard':
                ################################################################
                #
                # create a keyboard and the associated effort matrix
                #
                print_debug(1, 'loading keyboard from', config.get('kb_definition', 'keyboard_input'))
                keyboard = create_keyboard(config.get('kb_definition', 'keyboard_input'))
            case 'reporttriads':
                print_debug(1, 'reporting triad frequency')
                report_triads(keytriads, keyboard)
            case 'reportwordeffort':
                print_debug(1, 'reporting word efforts')
                report_word_effort(keyboard)
            case 'drawinputkeyboard':
                print_debug(1, 'drawing input keyboard')
                draw_keyboard(keyboard, )

                draw_keyboard($keyboard,"$CONF{pngfile_keyboard_input}",{title=>"$CONF{keyboard_input} layout"});
                printkeyboard($keyboard);

            case 'drawoutputkeyboard':
                print_debug(1, 'drawing output keyboard')

                draw_keyboard($keyboard,"$CONF{pngfile_keyboard_output}",{title=>"Optimized layout, runid $CONF{runid}"});
                printkeyboard($keyboard);

            case s if s.startswith('reporteffort'):
                # calculate the canonical effort associated with the original
                # keyboard layout - the layout will be altered to try to minimize this
                report_option = s.removeprefix('reporteffort')
                print_debug(1, 'calculating effort')
                config['options']['memorize'] = 'no'
                report_keyboard_effort(keytriads, keyboard, report_option)
                config['options']['memorize'] = 'yes'
      } elsif ($action =~ /reporteffort(.*)/i) {
        # calculate the canonical effort associated with the original
        # keyboard layout - the layout will be altered to try to minimize this
        my $effort_canonical;
        printdebug(1,"calculating effort");
        $CONF{memorize} = 0;
        report_keyboard_effort($keytriads,$keyboard,$1);
        $CONF{memorize} = 1;
      } elsif ($action =~ "optimize") {
        # optimize the keyboard layout to decrease the effort
        my $timer = [gettimeofday];
        $keyboard = optimize_keyboard($keytriads,$keyboard);
        $timer = tv_interval($timer);
        printkeyboard($keyboard);
        print "Total time spent optimizing: $timer s\n";
      } elsif ($action =~ /(exit|quit)/) {
        exit;
      } else {
        die "cannot understand action $action";
      }
    }

    exit;

# %% execution

if __name__ == '__main__':
    main()