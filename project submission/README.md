# Simulating Clacker Balls (Bolas)
My final project for Theory of Machine Dynamics (ME-314) course at Northwestern University.

<p align="center">
  <img align="center" src="https://github.com/seanmorton2023/ME314/blob/master/project%20submission/media/demo_path_following.gif" width="50%">
</p>


## Table of Contents

- [Project Description](#project-description)
- [Requirements](#requirements)
- [Dynamic Simulation](#dynamic-simulation)


## Project Description

In this project, I simulated and animated the dynamics of a "clacker balls" toy where two balls, suspended from strings originating at the same point, collide with each other and move in free space.

<p align="center">
  <img align="center" src="https://github.com/seanmorton2023/ME314/blob/master/project%20submission/media/clacker_balls.png" height="25%">
</p>

When we model this system as 2 square blocks at the end of two independent strings, there are 4 state variables: *θ_1, θ_2, Φ_1, Φ_2*
 where *θ_1,2* are the angles of the strings from the central axis and *Φ_1,2* are the angles of rotation of the masses themselves. [this deviates somewhat from the original toy, which has fixed masses]. This is illustrated below.

<p align="center">
  <img align="center" src="https://github.com/seanmorton2023/ME314/blob/master/project%20submission/media/clacker_frames.png" width="50%">
</p>

Frames used to compute equations of motion:
- Frame S will be at a fixed location on screen
- Frame P will be at the top loop of the Bolas
- Frames E1, E2 will be at the ends of the strings
- Frames B1, B2 will be the frames associated with the center of each Bola as it rotates.
- Lastly, an additional frame is used for trajectory following of the Bolas. Frame P is "forced" towards the position of this frame using PD control.


## Requirements

1. involve at least two bodies and be more than 2 but not more than 5 degrees of freedom (unless something makes the extra degrees of freedom straight forward);
1. include rotational inertia in at least one body;
1. include impacts;
1. include some sort of external forcing (but this could be friction–not necessarily control forces/torques).
1. be planar; we will be limiting ourselves to projects in 2D;
1. animate the resulting simulation to show that it “works”;
1. no spheres—anything like a “ball” must be modeled as a polygon (typically a triangle or rectangle) so that impacts have some sort of nontrivial updates.


## Dynamic Simulation

To simulate the dynamic system, I wrote symbolic expressions for the Euler-Lagrange equations of energy of the system, then solved them in real time using the SymPy library. Once the equations of motion are solved, linear algebra computes what the necessary updates to the Bolas' current position should be. This uses the Numpy library.

Energy is added to the system by the user, who can drag around the top loop of the Bolas to make the balls swing more energetically. This is done using interactions with a GUI made in Tkinter.

________

Demo #1: Simulation in real-time with the top of the toy following a fixed trajectory.

<p align="center">
  <img align="center" src="https://github.com/seanmorton2023/ME314/blob/master/project%20submission/media/demo_path_following.gif" width="50%">
</p>

________

Demo #2: Simulation in real-time with the top of the toy experiencing a force to follow the position of the user's cursor.

<p align="center">
  <img align="center" src="https://github.com/seanmorton2023/ME314/blob/master/project%20submission/media/demo_user_inputs.gif" width="50%">
</p>
