
try:
    import pygame, sys
    import tkinter
    from tkinter import messagebox
    import numpy as np
    import tensorflow as tf
    import matplotlib.pyplot as plt
except:
    exit(1)



#   Tämä classin avulla suoritetaan piirtäminen ja piirroksen pikseleiden tallentaminen listaan

class drawNum(object):

    def __init__(self, screen):
        
        self.screen = screen
        self.drawPos = pygame.PixelArray(screen)
        pygame.display.set_caption('Piirrä numero')
        screen.fill([0,0,0]) 
        pygame.init()
        pygame.display.update()

    
    # Piirtäminen
    def drawMouseCD(self):
        position = pygame.mouse.get_pos()
        click = pygame.mouse.get_pressed()
        if click[0] == True:
            
            pygame.draw.rect(self.screen, (255,255,255), (position[0], position[1], 20, 20))

            #   Skaalataan näytön pikselien määrä 28x28, koska neuroverkko opetettu sen kokoisilla kuvilla
            scaledScreen = pygame.transform.scale(self.screen, (28, 28))
            #   Otetaan pikselit listaan talteen ja käännetään lista oikein päin transpoosilla.
            self.drawPos = pygame.PixelArray(scaledScreen)
            self.drawPos = pygame.PixelArray.transpose(self.drawPos)
        return self.drawPos

    #   Kuvan ja listan tyhjentäminen
    def clearImage(self):
        
        self.screen.fill([0,0,0])
        self.drawPos = pygame.PixelArray(screen)
            
        
        
        
#   Tässä aliohjelmassa käyttäjän piirtämä kuva ajetaan neuroverkon läpi.
def Guess(X_image):
    mnist_model = tf.keras.models.load_model('mnistNN.model')
    
    #Muokataan kuvan data 3 dimensioiseksi matriisiksi ja normalisoidaan.
    X_image = np.reshape(X_image, (1, 28, 28))
    X_imageNorm = tf.keras.utils.normalize(X_image, axis=1)

    predictions = mnist_model.predict([X_imageNorm])
    # Predictionsiin tulee jokaiselle numerolle "todennäköisyys", valitaan niistä suurin.
    gNum = (np.argmax(predictions))
  
    #   Ponnahdusikkuna
    window = tkinter.Tk()
    window.withdraw()
    messagebox.showinfo("Arvaus", "Numero on: " + str(gNum))
    window.destroy()
    


   
#   Pyöritetään ohjelmaa ja reagoidaan hiiren painalluksiin.
def main(drawScreen):
    clock = pygame.time.Clock()
    run = True
    drawing = False
    while run:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False

            elif event.type == pygame.MOUSEBUTTONUP:
                
                
                drawing = False
                Guess(X_image)
                drawScreen.clearImage()

            elif drawing:
                X_image = drawScreen.drawMouseCD()
                
            
            
            elif event.type == pygame.MOUSEBUTTONDOWN:
                drawing = True
                
        pygame.display.flip()
        clock.tick(200)
            
          
    
    pygame.display.update()

# Luodaan screeni ja käynnistetään ohjelma. 
width = height = 560

screen = pygame.display.set_mode((width,height))

drawScreen = drawNum(screen)
main(drawScreen)
