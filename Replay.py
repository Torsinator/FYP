import sys
import os
import numpy as np
from PySide6.QtWidgets import QApplication, QWidget, QPushButton, QLabel, QVBoxLayout, QFileDialog, QMainWindow
from PySide6.QtCore import QTimer, QDir, QRect, Qt
from PySide6.QtGui import QPixmap, QPainter, QColor, QFont, QAction
from Game import Game

def read_all_objects(filename):
    objects = list()
    with open(filename, 'rb') as file:
        while True:
            try:
                obj = pickle.load(file)
                objects.append(obj)
            except EOFError:
                break
    return objects

class GridWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.rows = 0
        self.cols = 0
        
    def __init__(self, grid, rows, cols, cell_size=5):
        super().__init__()
        self.cell_size = cell_size
        self.grid = grid
        self.rows = rows
        self.cols = cols
        # self.setMinimumSize(self.cols * cell_size, self.rows * cell_size)

    def paintEvent(self, event):
        print("paintevent called")
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        print(f'w: {self.width()} h: {self.height()}')
        self.cell_size = int(min(self.width()/self.cols, self.height()/self.rows)) * 0.9

        # Draw grid
        for row in range(self.rows):
            for col in range(self.cols):
                rect = QRect(col * self.cell_size, row * self.cell_size, self.cell_size, self.cell_size)
                painter.setPen(Qt.black)
                painter.drawRect(rect)

                # Fill cell if marked
                if self.grid[row][col] != ".":
                    painter.setFont(QFont("Arial", 12))
                    painter.drawText(rect, Qt.AlignCenter, self.grid[row][col])

class ImagePlayer(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Image Frame Player")

        # Initialize variables
        self.log_file = ""
        self.lines = list()
        self.frames = []
        self.current_frame = 0

        # Create layout
        self.screen_layout = QVBoxLayout()
        
        # Create the menu bar
        menu_bar = self.menuBar()

        # Add a File menu
        file_menu = menu_bar.addMenu("File")

        open_action = QAction("Open", self)
        open_action.triggered.connect(self.open_log)
        file_menu.addAction(open_action)
        
        self.central_widget = QWidget(self)
        self.setCentralWidget(self.central_widget)

        # Create a grid widget
        self.grid_display = None
        self.play_button = None
        # layout.addWidget(self.grid_display)

        # Create open button
        # open_button = QPushButton("Open Log File")
        # open_button.clicked.connect(self.open_log)
        # layout.addWidget(open_button)
        
        self.central_widget.setLayout(self.screen_layout)

        # Set layout to the window

        # Set up timer
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)

    def open_log(self):
        log_file = QFileDialog.getOpenFileName(self, "Open Log File")
        if log_file:
            self.reset_layout()
            self.log_file = log_file[0]
            print(self.log_file)
            with open(self.log_file, 'r') as file:
                self.lines = file.read().splitlines()
            self.current_frame = 0
            self.load_grid()
            
            # Create play button
            self.play_button = QPushButton("Play")
            self.play_button.clicked.connect(self.play_frames)
            self.screen_layout.addWidget(self.play_button)
            
            self.load_frame(self.current_frame)

    def play_frames(self):
        if self.image_files:
            self.timer.start(100)  # Update frame every 100 milliseconds

    def update_frame(self):
        if self.image_files:
            self.current_frame += 1
            if self.current_frame >= len(self.image_files):
                self.current_frame = 0  # Loop back to the first frame
            self.load_frame(self.current_frame)

    def load_frame(self, frame_index):
        pass
        # self.play_button.setVisible(True)
        
    def load_grid(self):
        dimensions:str = self.lines[0]
        if not dimensions.startswith("0 GRID:"):
            raise ValueError("Malformed Log File")
        _, _, rows, cols = dimensions.split(" ")
        rows, cols = int(rows), int(cols)
        grid_data = self.lines[1:cols+1]
        loaded_array = np.loadtxt(grid_data, dtype=str)
        self.grid_display = GridWidget(loaded_array, rows, cols)
        self.screen_layout.addWidget(self.grid_display)
        
    def reset_layout(self):
        # Remove all widgets from the layout
        while self.screen_layout.count():
            item = self.screen_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        
def main():
    app = QApplication(sys.argv)
    player = ImagePlayer()
    player.showMaximized()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
    