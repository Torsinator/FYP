import Game

rows = 20
cols = 20
agents = 4
resources = 4


def main():
    game = Game(rows, cols, resources)
    game.generate()

if __name__ == '__main__':
    main()