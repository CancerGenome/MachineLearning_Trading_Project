import pandas as pd
import numpy as np
def draw_rectangle_forfirehydrant(center = [1,2], first = [0,1], second = [1,0],third = [5,4], forth = [4,5], height = 2, width = 2):
    # center means the center of firehydrant;
    # first, second, third, forth means the four point of rectangle, should be clockwise or anticlockwise
    # height and width is the size of new rectangle draw for fire hydrant

    center = [1,2.5]
    first = [0,1]
    second = [1,0]
    third = [5,4]
    forth = [4,5]
    height = 2
    width = 2
    slope = np.zeros(4)
    intercept = np.zeros(4)
    ### first check whether four points are clockwise or anticlockwise, whether this is the rectangle
    # will implement with for loop in future
    slope[0] = (second[1] - first[1])/(second[0] - first[0])
    slope[1] = (third[1] - second[1])/(third[0] - second[0])
    slope[2] = (forth[1] - third[1])/(forth[0] - third[0])
    slope[3] = (first[1] - forth[1])/(first[0] - forth[0])
    intercept[0] =  first[1] - slope[0] * first[0]
    intercept[1] =  second[1] - slope[1] * second[0]
    intercept[2] =  third[1] - slope[2] * third[0]
    intercept[3] =  forth[1] - slope[3] * forth[0]
    if(slope[0] * slope[1] != -1 or slope[1] * slope[2] != -1 ):
        print("wrong order of four points")
    #print(slope)
    #print(intercept)
    ### calculate the distance between center and each lines and get the closed line
    distance = np.zeros(4)
    for i in range(4):
        distance[i] = np.absolute(slope[i] * center[0] + intercept[i] - center[1]) /np.sqrt(slope[i]*slope[i] + 1) # https://en.wikipedia.org/wiki/Distance_from_a_point_to_a_line
    print(distance)
    print(np.argmin(distance))
    index = np.argmin(distance) # this is the index of closest line
    index_p = np.absolute(min_index - 2)# this is the index of parallele line
    ### check whether one side or the other
    intercept_center = center[1] - slope[index]*center[0]
    inside = np.zeros(1) # outside: 0, inside: 1
    if( (intercept_center - intercept[index])*(intercept_center - intercept[index_p]) <=0 ):
        inside = 1

if __name__ == "__main__":
    draw_rectangle_forfirehydrant()
