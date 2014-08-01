import cv2
import os

pos = [e for e in os.listdir('.') if e.startswith('snapshot')]
fg = cv2.imread('../../mouse/mouse_tran.png')
o = [(y,x) for y in range(fg.shape[0]) for x in range(fg.shape[1]) if fg[y][x].any() ]

def create_pos(x):
    try:
        bg[x[0]+1][x[1]+3] = fg[x[0]][x[1]]
    except IndexError:
        print(x)
    #print(x)

for i,bgfile in enumerate(pos):
    bg = cv2.imread(bgfile)
    #print(bg.shape)
    map(create_pos, o)
    filename = 'sample_{0:04}.png'.format(i)
    cv2.imwrite(filename, bg)


