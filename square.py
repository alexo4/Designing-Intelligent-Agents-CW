

class Square:

    def __init__(self, balls, player):
        self.balls = balls
        self.player = player


    def is_empty(self):
        if self.balls == 0 & self.player == False:
            return True
        return False

    def set_player(self, player):
        self.player = player

    def set_balls(self, balls):
        self.balls = balls

    def get_player(self):
        return self.player

    def get_balls(self):
        return self.balls