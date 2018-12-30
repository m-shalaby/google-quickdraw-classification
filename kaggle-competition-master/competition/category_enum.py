from enum import Enum

class CategoryEnum(Enum):
    sink=0
    pear=1
    moustache=2
    nose=3
    skateboard=4
    penguin=5
    peanut=6
    skull=7
    panda=8
    paintbrush=9
    nail=10
    apple=11
    rifle=12
    mug=13
    sailboat=14
    pineapple=15
    spoon=16
    rabbit=17
    shovel=18
    rollerskates=19
    screwdriver=20
    scorpion=21
    rhinoceros=22
    pool=23
    octagon=24
    pillow=25
    parrot=26
    squiggle=27
    mouth=28
    empty=29
    pencil=30

def get_category_index(category:str):
    for c in CategoryEnum:
        if str(c).split('.')[1] == category:
            return c.value
    return -1
    