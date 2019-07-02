# Game_of_life

### Introduction

The Game of Life, also known simply as Life, is a cellular automaton devised by the British mathematician John Horton Conway in 1970.

The game is a zero-player game, meaning that its evolution is determined by its initial state, requiring no further input. One interacts with the Game of Life by creating an initial configuration and observing how it evolves.

The purpose of this project is to simulate the game of life through a 2D matrix, observing the evolution of life under specific configuration of matrix. As stated before, the main mechanism of this game is to update the life state generation by generation, which requires continuous computation and update.

Thus, this project  implemented three different methods including two serial approaches and one parallel approach to achieve this game, then made a comparison between these approaches.



### Game description

In this project, game of life was simulated by a 2D matrix, where each cell represents a unit with two kinds of state: live or dead. As shown below, each cell has 8 neighbors at most, the next state of each cell is determined by the number of live neighbors around.

Here is the rules of the game:

1. Any live cell with fewer than two live neighbors dies, as if by underpopulation
2. Any live cell with two or three live neighbors lives on to the next generation.
3. Any live cell with more than three live neighbors dies, as if by overpopulation.
4. Any dead cell with exactly three live neighbors becomes a live cell, as if by reproduction.
