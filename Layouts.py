import pygame
class Layout:
    def __init__(self):
        x,y = 0,1 # Position index
        #Basis
        self.fps = 30
        self.width = 1240
        self.height = 720
        self.resize = 640
        self.edit1Idx = 1
        self.edit2Idx = 2
        #Contorls
        #Mousecode
        self.leftButton = 1
        self.centerButton = 2
        self.backWheel = 5
        self.upWheel = 4
        #Colors
        self.black = (0, 0, 0)
        self.white = (255, 255, 255)
        self.red = (255, 0, 0)
        #Apps
        self.edit1Pos = ((900, 110), (1150, 150))  # x1,y1 x2,y2
        self.edit2Pos = ((955, 160), (1100, 200))  # x1,y1 x2,y2
        self.font = pygame.font.SysFont('arial', 50, True, True)
        self.fontSmall = pygame.font.SysFont('arial', 30, True, True)

        self.label1 = [700, 40] # Image label pos
        self.label2 = [694, 90] # Isoverlap label pos
        self.label3 = [688, 140] # Numoverlap label pos
        self.label4 = [687, 250] # Positions label pos

        self.edit1 = [920, 90] # IsOverlapVal edit pos
        self.edit2 = [974, 145] # OverlapNumval edit pos
        self.edit3 = [974, 300]  # Position edit pos

        self.cursorEdit1 = ((self.edit1[x] - 10, self.edit1[y]+ 15 * 1), (self.edit1[x] - 10, self.edit1[y] + 25 * 2))
        self.cursorEdit2 = ((self.edit2[x] - 10, self.edit2[y] + 15 * 1), (self.edit2[x] - 10, self.edit2[y] + 25 * 2))
        # Position list stand pos
        self.standPos = (700, 320)