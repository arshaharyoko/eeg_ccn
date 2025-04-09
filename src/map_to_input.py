import pynput
import time

class InputPeripherals:
    def __init__(self):
        self.keyboard = pynput.keyboard.Controller()
        self.mouse = pynput.mouse.Controller()
        self.x, self.y = self.mouse.position
    
    def cursor_translate(self, x, y):
        """
        Translate and update cursor position by x and y
        Parameters:
            (x, y): Signed integer
        """
        self.mouse.position = (self.x+x, self.y+y)
        self.x, self.y = self.mouse.position

    def wasd_map(self, key):
        if key==1:
            self.keyboard.press('W')
        elif key==2:
            self.keyboard.press('S')
        elif key==3:
            self.keyboard.press('A')
        elif key==4:
            self.keyboard.press('D')