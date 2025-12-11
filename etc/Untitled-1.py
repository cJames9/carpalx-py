# %%
# 
import os
os.chdir(os.environ['OneDrive'])
os.chdir('Documentos//GitHub//qmk_firmware')
import json
from PIL import Image, ImageDraw, ImageFont, ImageColor

file = './/keyboards//sofle_choc//keyboard.json'

j = json.load(open(file))

kb = j['layouts']['LAYOUT']['layout']


def draw_keyboard(keyboard, file, parameters):
    '''
    Create an image of the keyboard
    '''
    # TODO: review row positioning

    file = self.resolve_path(file)
    
    imageparamset = self.config.getint('kb_parameters', 'imageparamset')
    image_params = self.config[f'imageparamsetdef {imageparamset}']
    imagedetaillevel = self.config.getint('kb_parameters', 'imagedetaillevel')
    image_detail = self.config[f'imagedetaildef {imagedetaillevel}']
    colors = dict(self.config[f'color {imageparamset}'])
    
    keysize = 23
    charxshift = 3
    ucyshift = 11
    lcyshift = 9
    keymargin = 0.2
    keymargin = keymargin * keysize if keymargin < 1 else keymargin
    shadow = 1
    
    width = int(round((20) * (24), 0))
    height = int(round(8 * (24) + 15, 0))
    
    image = Image.new(mode='RGBA', size=(width, height), color=(255, 255, 255, 0))
    draw = ImageDraw.Draw(image)
    
    draw.rectangle([0, 0, width, height], fill='white')
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
    
    # %%

    width = int(round((20) * (24), 0))
    height = int(round(8 * (24) + 15, 0))
    
    image = Image.new(mode='RGBA', size=(width, height), color=(255, 255, 255, 0))
    draw = ImageDraw.Draw(image)
    
    draw.rectangle([0, 0, width, height], fill='white')

    for key in kb:
        key_h = (key['h'] - 1) * keysize if 'h' in key else 0
        key_x = (key['x'] + 1) * keymargin + key['x'] * keysize
        key_y = (key['y'] + 1) * keymargin + key['y'] * keysize

        draw.rectangle([key_x, key_y - key_h / 2, key_x + keysize, key_y + keysize + key_h / 2], fill='white', outline='black')
        
    image.show()
    
    # %%
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