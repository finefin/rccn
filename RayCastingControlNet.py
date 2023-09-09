'''

Info:  [TODO]



'''

import numpy as np
import multiprocessing
import pygame as pg
import io
import sys
import cv2 as cv
import base64
import json
import requests
from PIL import Image
from numba import njit
from threading import Thread

scrCount = 1
theSeed = 1543888376
saveIncremental = True # saves the images in project folder 
saveCnet = False
cnetImgScreen = pg.Surface((512, 512))
cnetImg = pg.Surface((1024, 512))
lImg = pg.Surface((1024, 512))

objects = []  # buttons

def encodeImage ( img, width=512, height=512 ):
    string_image = pg.image.tostring(img, 'RGB')
    temp_surf = pg.image.fromstring (string_image,(width, height),'RGB' )
    tmp_arr = pg.surfarray.array3d(temp_surf)            
    tmp_arr = cv.transpose(tmp_arr) # transpose or image is x/y-flipped
    retval, bytes = cv.imencode('.jpg', tmp_arr)
    encodedImage = base64.b64encode(bytes).decode('utf-8')        
    return encodedImage 
# encode the init images


#print(Image.open("mask.jpg"))
fImg = pg.image.load('init-1024-w.jpg')
firstImage = encodeImage ( fImg , 1024, 512 )
latentMask = encodeImage ( pg.image.load('mask.jpg'), 1024, 512 )
lastImage = firstImage
lImg.blit(fImg, (0,0))




class Button():
    def __init__(self, x, y, width, height, key, buttonText='Button', onclickFunction=None, onePress=False):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.onclickFunction = onclickFunction
        self.onePress = onePress
        self.key = key

        self.fillColors = {
            'normal': '#ffffff',
            'hover': '#666666',
            'pressed': '#333333',
        }

        self.buttonSurface = pg.Surface((self.width, self.height))
        self.buttonRect = pg.Rect(self.x, self.y, self.width, self.height)

        font = pg.font.SysFont("Arial", 10)
        self.buttonSurf = font.render(buttonText, True, (20, 20, 20))

        self.alreadyPressed = False

        objects.append(self)

    def process(self, screen):

        mousePos = pg.mouse.get_pos()
        
        self.buttonSurface.blit(self.buttonSurf, [
            self.buttonRect.width/2 - self.buttonSurf.get_rect().width/2,
            self.buttonRect.height/2 - self.buttonSurf.get_rect().height/2
        ])
        screen.blit(self.buttonSurface, self.buttonRect)
        
        self.buttonSurface.fill(self.fillColors['normal'])
        if self.buttonRect.collidepoint(mousePos):
            self.buttonSurface.fill(self.fillColors['hover'])

            if pg.mouse.get_pressed(num_buttons=3)[0]:
                self.buttonSurface.fill(self.fillColors['pressed'])

                if self.onePress:
                    self.onclickFunction()
                   
                elif not self.alreadyPressed:
                    self.onclickFunction()
                    self.alreadyPressed = True
                return self.key

            else:
                self.alreadyPressed = False

# button callback functions that we don't really need 
def btnGo():
    print('-> go')#  
    
def btnLeft():
    print('-> left')
    
def btnRight():
    print('-> right')
    
def main():

    global objects
    global lastImage
    global firstImage
    global cnetImg
    
    size = 100 # size of the map
    posx, posy, posz = (1, np.random.randint(1, size -1), 0.5)
    rot, rot_v = (0, 0.1)
    lx, ly, lz = (size/2-0.5, size/2-0.5, 1)    
    mapc, maph, mapr, exitx, exity = maze_generator(posx, posy, size)
    res, res_o = 3, [64, 96, 112, 160, 192, 224, 448]
    width, height, mod, inc, sky, floor = adjust_resol(res_o[res])
   
    nuc = 8
    pool = multiprocessing.Pool(processes = nuc)
    renderFrame = True
    alphaSDImage = 128
    
    bench = []
    running = True
    pg.init()
    # font = pg.font.SysFont("Arial", 10)
    screen = pg.display.set_mode((1024, 512))

    clock = pg.time.Clock()
    pg.mouse.set_visible(True)
    # pg.mouse.set_pos([400, 300])

    imgAlpha = 0  
    blitDelay = 0    
    
    goButton = Button(226, 480, 80, 40, 'go', 'Go', btnGo)
    leftButton = Button(126, 480, 80, 40, 'left', 'Turn', btnLeft)
    rightButton = Button(326, 480, 80, 40, 'right', 'Turn', btnRight)

    while running:
        
        for event in pg.event.get():
            if event.type == pg.QUIT:
                running = False
            if event.type == pg.KEYDOWN:
                if event.key == pg.K_ESCAPE:
                    running = False
                if event.key == ord('f'):              
                    alphaSDImage = 0 
                if event.key == ord('g'):              
                    alphaSDImage = 127
                if event.key == ord('h'):              
                    alphaSDImage = 255                    
                if event.key == ord('p'):              
                    renderFrame = True                        
                if event.key == ord('q'): # change resolution
                    if res > 0 :
                        res = res-1
                        width, height, mod, inc, sky, floor = adjust_resol(res_o[res])
                if event.key == ord('e'):
                    if res < len(res_o)-1 :
                        res = res+1
                        width, height, mod, inc, sky, floor = adjust_resol(res_o[res])
                        
        param_values = []
        for j in range(height): #vertical loop 
            rot_j = rot_v + np.deg2rad(24 - j/mod)
            for i in range(width): #horizontal vision loop
                param_values.append([rot, i, j, inc, rot_j])
        tam = len(param_values)
        lista = []
        pixels = []
        
        for i in range(nuc):
            lista.append([i, param_values[i*int(tam/nuc):(i+1)*int(tam/nuc)],
                          mapc, maph, lx, ly, lz, exitx, exity, mapr, posx, posy, posz, mod])

        retorno = pool.map(caster, lista)

        for i in range(nuc):
            pixels.append(retorno[i][1])

        pixels = np.reshape(pixels, (height,width,3))
        pixels = np.asarray(pixels)/np.sqrt(np.max(pixels))
        
        

        
        for object in objects:
            key = object.process(screen)
            if key == "go":
                et = 0.5 #clock.tick()/500
                posx, posy = (posx + et*np.cos(rot), posy + et*np.sin(rot))
                renderFrame = True 
            if key == "left":
                rot = rot + 0.2
                renderFrame = True 
            if key == "right":
                rot = rot - 0.2
                renderFrame = True               
           
        
        
        # player's movement
        if (int(posx) == exitx and int(posy) == exity):
            break

        pressed_keys = pg.key.get_pressed()        
        posx, posy, rot, rot_v = keyboardMovement(pressed_keys, posx, posy, rot, rot_v, maph, clock.tick()/500)       
        
        cnetImgScreen = pg.surfarray.make_surface((np.rot90(pixels*255)).astype('uint8'))
        cnetImgScreen = pg.transform.scale(cnetImgScreen, (512, 512))
        
        # pg.draw.rect(cnetImg, (0,0,0), pg.Rect(0, 0, 1024, 512))      
        # cnetImg.blit(cnetImgScreen, (512,0))
        
        pg.Surface.set_alpha(lImg, alphaSDImage)
                        
        screen.blit(cnetImgScreen, (512, 0) )         
        screen.blit(lImg, (0,0) )


        pg.display.flip()

        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
        if renderFrame == True:
            blitDelay = blitDelay + 1
            delayTrigg = round(clock.get_fps()) # wait for next frame after blit
            if blitDelay >= delayTrigg:
                blitDelay = 0
                renderFrame = False               
                encoded_image = encodeImage ( cnetImgScreen, 512, 512 ) # image for controlnet
                encoded_sd_image = encodeImage ( lImg, 1024, 512 ) # will be stored as global lastImage
                thread = Thread(target=sendCnetImage, args=(encoded_image,encoded_sd_image))
                thread.start()
            
            
            
    stop_thread = True
    pg.quit()
    pool.close()




    
# run this in a thread or program will pause until response of SD
def sendCnetImage (encodedImage, encodedSdImage):
    global theSeed
    global lImg         # the surface that shows the sd image
    global cnetImg      # the surface that shows the controlnet image
    global saveIncremental
    global saveCnet
    global scrCount  
    global lastImage
    global firstImage
    
    print ("encodedSdImage ---->")
    print (encodedSdImage)
    print ("<----")
    
    print ("latentMask ---->")
    print (latentMask)
    print ("<----")
    
    print ("encodedImage ---->")
    print (encodedImage)
    print ("<----")
    
    url = "http://127.0.0.1:7860"
    payload = {
        "prompt": 'a cardboard (hallway:1.2), (flat floor:1.4), digital art',
        "negative_prompt": "",
        "init_images": [encodedSdImage],
        "mask": latentMask,
        "mask_blur": 0,
        "sampler_index": "UniPC",
        "seed": theSeed,
        "denoising_strength": 0.9,
        "batch_size": 1,
        "steps": 8,
        "cfg_scale": 6,
        "alwayson_scripts": {
            "Additional networks for generating":{
                "args":[True, False,'LoRA', 'None', 1, 1, 'LoRA', 'concept/more_details(3b8aa1d351ef)', -2, -2, 'LoRA', 'None', 1, 1, 'LoRA', 'None', 1, 1, 'LoRA', 'None', 1, 1]
                },
            "controlnet": {
                "args": [
                    {
                        "input_image": encodedImage,
                        'enabled': True,
                        'module': "depth_midas",
                        'model': "control_v11f1p_sd15_depth [cfd03158]",
                        'weight': 1,
                        'resize_mode': 1,
                        'low_vram': False,
                        'processor_res': 512,
                        'threshold_a': 0,
                        'threshold_b': 1,
                        'guidance_start': 0, 
                        'guidance_end': 1, 
                        'control_mode': 0,
                        'pixel_perfect': True
                    },
                    {
                        "input_image": firstImage,
                        "model": "",
                        "module": "reference_adain+attn",
                        "weight": 0.8,
                        'enabled': True,
                        'pixel_perfect': True, 
                        'control_mode': 0
                    }
                ]
            }
        }
    }
    
    # Trigger Generation
    response = requests.post(url=f'{url}/sdapi/v1/img2img', json=payload)
    # Read results
    r = response.json()
    result = r['images'][0]   
    renderFrame = False
    lastImage = encodedSdImage
    lImgTmp = pg.image.load( io.BytesIO  (base64.b64decode(result.split(",", 1)[0]) ) )
    lImg.blit(lImgTmp, (0,0) )
    #pg.display.flip()
   
 #   if scrCount == 1:
 #       firstImage = encodedSdImage
    
    if saveIncremental == True:
        pg.image.save(lImg, "scr_%d.jpg" % scrCount )
        if saveCnet == True:
            pg.image.save(cnetImg, "_out/scrCnet_%d.jpg" % scrCount )
        
    scrCount += 1
    
def maze_generator(x, y, size):
    mapc = np.random.uniform(0, 0.1, (size,size,3)) 
    mapr = np.random.choice([0, 0, 0, 0], (size,size))
    maph = np.random.choice([0, 0, 0, 0, 0, 0, .9, .9, .9], (size,size))
    maph[0,:], maph[size-1,:], maph[:,0], maph[:,size-1] = (.85,.85,.85,.85)

    mapc[x][y], maph[x][y], mapr[x][y] = (0, 0, 0)
    count = 0 
    while 1:
        testx, testy = (x, y)
        if np.random.uniform() > 0.5:
            testx = testx + np.random.choice([-1, 1])
        else:
            testy = testy + np.random.choice([-1, 1])
        if testx > 0 and testx < size -1 and testy > 0 and testy < size -1:
            if maph[testx][testy] == 0 or count > 5:
                count = 0
                x, y = (testx, testy)
                mapc[x][y], maph[x][y], mapr[x][y] = (0, 0, 0)
                if x == size-2:
                    exitx, exity = (x, y)
                    break
            else:
                count = count+1
    return mapc, maph, mapr, exitx, exity

def keyboardMovement(pressed_keys,posx, posy, rot, rot_v, maph, et):
    
    x, y = (posx, posy)
    
    # p_mouse = pg.mouse.get_pos()
    # rot = rot + 4*np.pi*(0.5-(p_mouse[0]-400)/150000)
    rot_v = -0.1 # vertical view angle
    
    if pressed_keys[pg.K_UP] or pressed_keys[ord('w')]:
        x, y = (x + et*np.cos(rot), y + et*np.sin(rot))
        
    if pressed_keys[pg.K_DOWN] or pressed_keys[ord('s')]:
        x, y = (x - et*np.cos(rot), y - et*np.sin(rot))
        
    if pressed_keys[pg.K_LEFT] or pressed_keys[ord('a')]:
        # x, y = (x - et*np.sin(rot), y + et*np.cos(rot))
        rot = rot + 0.1
        
    if pressed_keys[pg.K_RIGHT] or pressed_keys[ord('d')]:
        #x, y = (x + et*np.sin(rot), y - et*np.cos(rot))
        rot = rot - 0.1
        
    if maph[int(x)][int(y)] == 0:
        posx, posy = (x, y)
        
    
                                                
    return posx, posy, rot, rot_v
        
       
@njit(fastmath=True)
def fast_ray(x, y, z, cos, sin, sinz, maph):
    while 1:
        x, y, z = x + cos, y + sin, z + sinz
        if (z > 1 or z < 0):
            break
        if maph[int(x)][int(y)] > z:
            break        
    return x, y, z        

def view_ray(x, y, z, cos, sin, sinz, mapc, lx, ly, lz, maph, exitx, exity):
    
    x, y, z = fast_ray(x, y, z, cos, sin, sinz, maph)
    dtol = np.sqrt((x-lx)**2+(y-ly)**2+(lz-1)**2)

    if z > 1: # ceiling
##        c = np.asarray([0.3,0.7,1])
        c = np.asarray([.6,.6,.6]) 
    elif z < 0: # floor
        c = np.asarray([1,1,1])
    else:
        c = np.asarray([1,1,1]) # if all fails

    h = 2*np.clip(1/dtol, 0, 1)
    c = c*h
    return c, x, y, z, dtol

@njit(fastmath=True)
def shadow_ray(x, y, z, lx, ly, lz, maph, c, inc, dtol):
    dx, dy, dz = inc*5*(lx-x)/dtol, inc*5*(ly-y)/dtol, inc*5*(lz-z)/dtol
    mod = 1
    while 1:
        x, y, z = (x + dx, y + dy, z + dz)
        if maph[int(x)][int(y)]!= 0 and z<= maph[int(x)][int(y)]:
            mod = mod*0.9
            if mod < 0.5:
                break
        elif z > 0.9:
            break
    return c*mod

def reflection(x, y, z, cos, sin, sinz, mapc, lx, ly, lz, maph, exitx, exity, c, posz, inc, mapr, recur):
    if abs(z-maph[int(x)][int(y)])<abs(sinz):
        sinz = -sinz
    elif maph[int(x+cos)][int(y-sin)] != 0:
        cos = -cos
    else:
        sin = -sin
    c2, x, y, z, dtol = view_ray(x, y, z, cos, sin, sinz, mapc, lx, ly, lz, maph, exitx, exity)
    if z < 1:
        c2 = shadow_ray(x, y, z, lx, ly, lz, maph, c2, inc, dtol)
    if (mapr[int(x)][int(y)] != 0 and z < 1 and z > 0 and not recur):
        c2 = reflection(x, y, z, cos, sin, sinz, mapc, lx, ly, lz, maph, exitx, exity, c2, posz, inc, mapr, recur=True)
    c = (c + c2)/2
    return c

def caster(lista):
    param_values = lista[1]
    mapc = lista[2]
    maph = lista[3]
    lx = lista[10] # put the light on player position
    ly = lista[11]
    lz = lista[12] - 1
    exitx = lista[7]
    exity = lista[8]
    mapr = lista[9]
    posx = lista[10]
    posy = lista[11]
    posz = lista[12]
    mod = lista[13]
    
    pixels = []
    
    for values in param_values:
        rot = values[0]
        i = values[1]
        j = values[2]
        inc = values[3]
        rot_j = values[4]
        rot_i = rot + np.deg2rad(i/mod - 30)
        x, y, z = (posx, posy, posz)
        sin, cos,  = (inc*np.sin(rot_i), inc*np.cos(rot_i))
        sinz = inc*np.sin(rot_j)
        c, x, y, z, dtol = view_ray(x, y, z, cos, sin, sinz, mapc, lx, ly, lz,
                                    maph, exitx, exity)
        if z < 1:
            c = shadow_ray(x, y, z, lx, ly, lz, maph, c, inc, dtol)
            #if mapr[int(x)][int(y)] != 0 and z > 0:
            #    c = reflection(x, y, z, cos, sin, sinz, mapc, lx, ly, lz, maph,
            #                   exitx, exity, c, posz, inc, mapr, recur=False)
                    
        pixels.append(c)

    return lista[0], pixels

def ray_caster(x, y, i, ex, ey, maph, mapc, sin, cos, n, half, mod):
    zz= 0.5
    if half == None:
        zz = 0.1
    x, y, n, tc, ty = fast_ray_caster(x, y, zz, cos, sin, maph, n, i, ex, ey, mod)
    h , c = shader(n, maph, mapc, sin, cos, x, y, i, mod)
    
    if maph[int(x)][int(y)] < 0.5 and half == None:
        half = [h, c, n]
        x, y, n, tc2, ty2 = fast_ray_caster(x, y, 0.5, cos, sin, maph, n, i, ex, ey, mod)
        ty, tc = ty + ty2, tc + tc2
        h , c = shader(n, maph, mapc, sin, cos, x, y, i, mod)
    return(c, h, x, y, n, half, ty, tc)


@njit(fastmath=True)
def fast_ray_caster(x, y, z, cos, sin, maph, n, i, ex, ey, mod):
    ty, tc = [], []
    while 1:
        n = n+1
        x, y = x + cos, y + sin
        if z < 0.5 and int(x*2)%2 == int(y*2)%2:
                th = 1/(0.05/mod * n)#*np.cos(np.deg2rad(i/mod - 30)))
                if th < 1  and th >= 0:
                    ty.append(th)
                    if int(x) == ex and int(y) == ey:
                        tc.append(np.asarray([0,0,1]))
                    else:
                        tc.append(np.asarray([0,0,0]))
        if maph[int(x)][int(y)] > z:
            break        
    return x, y, n, tc, ty

def shader(n, maph, mapc, sin, cos, x, y, i, mod):
    
    h = np.clip(1/(0.05/mod * n), 0, 1)#*np.cos(np.deg2rad(i/mod-30))), 0, 1)
    c = np.asarray(mapc[int(x)][int(y)])*(0.4 + 0.6 * h)
    
    if maph[int(x+cos)][int(y-sin)] == 1:
        c = 0.85*c
        
        if maph[int(x-cos)][int(y+sin)] == 1 and sin >0:
            c = 0.7*c
    return h, c

def reflection_caster(x, y, i, ex, ey, maph, mapc, sin, cos, n, c, h, half, pixels, ty, tc, height, mod):
    
    hor = int(height/2)
    hh = int((h*height)/2)
    pixels[hor-hh:hor+hh,i] = np.add(pixels[hor-hh:hor+hh,i], np.asarray([c]*(hh*2)))/2
    
    if maph[int(x+cos)][int(y-sin)] > 0.5:
        cos = -cos
        
    else:
        sin = -sin
        
    c2, h2, x, y, n2, half2, ty2, tc2 = ray_caster(x, y, i, ex, ey, maph, mapc, sin, cos, n, half, mod)
        
    ty, tc = ty + ty2, tc + tc2
    hh = int((h2*height)/2)
    pixels[hor-hh:hor+hh,i] = (c + c2)/2
    
    if half2 != None and half == None:
        hh = int((half2[0]*height)/2)
        pixels[hor:hor+hh,i] = (c + half2[1])/2
        
    elif half != None:
        hh = int((half[0]*height)/2)
        pixels[hor:hor+hh,i] = half[1]
           
    return pixels, ty, tc     

def adjust_resol(width):
    height = width
    mod = width/64
    inc = 0.05/mod
    gradient = np.linspace(0,1,int(height/2-1))
    sky = np.asarray([gradient,gradient,gradient]).T
    floor = np.asarray([gradient,gradient,gradient]).T
    print('Resolution: ', width, height)
    return width, height, mod, inc, sky, floor

if __name__ == '__main__':
    main()
    
    
