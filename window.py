import tkinter as tk

class Window:

    def __init__(self, court_size):
        self.canvas = None
        self.court = None
        self.court_size = court_size
        self.root = tk.Tk()
        self.window_size = 400
        self.square_size = self.window_size / court_size
        self.initialise(self.root)
        self.root.update()
        #self.root.mainloop()
        self.objects = {}

    def initialise(self, window):
        window.resizable(False, False)
        self.canvas = tk.Canvas(window, width=1000, height=1200)
        self.canvas.pack()

        self.canvas.create_rectangle(296, 192, 704, 1008,
                                     outline="purple", width=8)

        self.canvas.create_rectangle(300,596, 699, 603, outline="purple", width=1, fill="purple")

        for i in range(0, self.court_size * 2):
            for j in range(0, self.court_size):

                y_offset = 0
                if i > self.court_size - 1:
                    y_offset = 8

                self.canvas.create_rectangle(300 + (self.square_size * j), 196 + y_offset + (self.square_size * i),
                                        300 + (self.square_size * j) + self.square_size,
                                        196 + y_offset + (self.square_size * i) + self.square_size)



    def initialise_players(self, court):

        self.court = court

        for cv_object in self.objects.values():
            self.canvas.delete(cv_object)

        self.objects = {}

        player_count = 0
        print("initialise")

        for i in range(0, len(self.court)):
            for j in range(0, len(self.court[i])):

                y_offset = 0
                if i > len(self.court[i]) - 1:
                    y_offset = 8

                if self.court[i][j].get_player():
                    player = self.draw_player(self.court[i][j].get_player(), y_offset)
                    self.objects.update({"player%d" % player_count : player})
                    player_count += 1

                if self.court[i][j].get_balls() > 0:
                    self.draw_balls(j, i, y_offset, self.court[i][j].get_balls())




    def update_player(self, player, x, y):

        print(str(x) + " " + str(y))
        x = x * self.square_size
        y = y * self.square_size
        self.canvas.move(self.objects["player%d" % player], x, y)
        #self.root.update()
        print(self.canvas.find_withtag('text'))
        self.canvas.delete('text')
        #print(self.canvas.index(cv_list[0]))

    def remove_player(self, player):
        self.canvas.delete(self.objects["player%s" % player])

    def update_balls(self):

        self.canvas.delete('balls')

        for i in range(0, len(self.court)):
            for j in range(0, len(self.court[i])):

                y_offset = 0
                if i > len(self.court[i]) - 1:
                    y_offset = 8

                if self.court[i][j].get_balls() > 0:
                    self.draw_balls(j, i, y_offset, self.court[i][j].get_balls())



    def draw_player(self, player, y_offset):

        return self.canvas.create_oval(300 + (self.square_size * player.x) + self.square_size / 4,
                           196 + y_offset + (self.square_size * player.y) + self.square_size / 4,
                           300 + (self.square_size * player.x) + self.square_size * 0.75,
                           196 + y_offset + (self.square_size * player.y) + self.square_size * 0.75,
                           fill=player.team)


    def draw_balls(self, x, y, y_offset, balls):

        if y_offset == 0:
            y_offset = self.square_size - 12

        self.canvas.create_text(300 + (self.square_size * x) + self.square_size - 4,
                                196 + y_offset + (self.square_size * y) + 6, text=str(balls),font='Helvetica 8 bold',
                                tag='balls')

        return self.canvas.create_oval(300 + (self.square_size * x) + 2,
                                       196 + y_offset + (self.square_size * y) + 2,
                                       300 + (self.square_size * x) + 10,
                                       196 + y_offset + (self.square_size * y) + 10,
                                       fill="green", tag='balls')


    def start_mainloop(self):
        self.root.mainloop()