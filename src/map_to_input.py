import pynput
# import time

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

    def asd_map(self, key):
        if key==2:
            self.keyboard.press('s')
            self.keyboard.release('s')
        elif key==1:
            self.keyboard.press('b')
            self.keyboard.release('b')
        elif key==3:
            self.keyboard.press('a')
            self.keyboard.release('a')
        elif key==4:
            self.keyboard.press('d')
            self.keyboard.release('d')

    def _map(self, key):
        self.keyboard.press(key)
        self.keyboard.release(key)