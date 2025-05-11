class Player:

    def __init__(self, name, x, y, team):
        self.name = name
        self.x = x
        self.y = y
        self.team = team
        self.alive = 1
        self.throw_cooldown = 0
        self.move_cooldown = 0
        self.dodge_cooldown = 0
        self.incoming_ball = 0

    def get_team(self):
        return self.team

    def set_alive(self, alive):
        self.alive = alive

    def get_alive(self):
        return self.alive

    def get_x(self):
        return self.x

    def set_x(self, x):
        self.x = x

    def get_y(self):
        return self.y

    def set_y(self, y):
        self.y = y

    def get_throw_cooldown(self):
        return self.throw_cooldown

    def set_throw_cooldown(self, cooldown):
        self.throw_cooldown = cooldown

    def get_move_cooldown(self):
        return self.move_cooldown

    def set_move_cooldown(self, cooldown):
        self.move_cooldown = cooldown

    def get_dodge_cooldown(self):
        return self.dodge_cooldown

    def set_dodge_cooldown(self, cooldown):
        self.dodge_cooldown = cooldown

    def get_incoming_ball(self):
        return self.incoming_ball

    def set_incoming_ball(self, ball):
        self.incoming_ball = ball

    def get_name(self):
        return self.name