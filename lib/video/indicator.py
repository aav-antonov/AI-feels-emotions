import argparse
import json
import os





from PIL import Image, ImageDraw
from PIL import ImageFont

## specifies color of indicator bar which depends on the strength of the siganl
def value_to_color(value,value_max):

    if value   <= (1/5)*value_max: line_color = (102, 3, 252)#"blue"
    elif value <= (2/5)*value_max: line_color = 3, 157, 252#""blue2""
    elif value <= (3/5)*value_max: line_color = (231, 252, 3)#"yellow"
    elif value <= (4/5)*value_max: line_color = (252, 123, 3)#"orange"
    else: line_color = (252, 15, 3) #"red"
    return line_color


class IndicatorOnImage:

    """
    IndicatorOnImage: insert indicator bar on image
    BarNames = {0: "Anger", 1: "Contempt", 2: "Disgust", 3: "Fear", 4: "Happiness", 5: "Neutral", 6: "Sadness",  7: "Surprise"}
    #indicator_values:  dictinary: int -> int
    """

    def __init__(self,  BarNames, Title=None, scale=1, position = "LL", value_max = 10 ):
        self.value_max = value_max
        self.Title = Title
        self.BarNames = BarNames # dictinary: int ->string
        self.N = len(self.BarNames)
        self.size_Title_H = 0
        if self.Title != None: self.size_Title_H = int(20 * scale)

        self.size = (int(150 * scale), int(15 * self.N * scale + self.size_Title_H))
        self.size_line_H = int(15 * scale)
        self.TitleFont = int(14 * scale)
        self.BarFont = int(10 * scale)

    def add_on_image(self, file_image, indicator_values, output_path = None):
        #indicator_values:  dictinary: int -> int
        if output_path == None: output_path = file_image

        image = Image.open(file_image)
        draw = ImageDraw.Draw(image)
        (W, H) = image.size

        draw.rectangle((0,  H - self.size[1], self.size[0], H), fill="black")


        for i, id in enumerate(indicator_values):
            (x,y, x1,y1) = (0, H - self.size[1] + self.size_Title_H + i * self.size_line_H +10) + (40, H - self.size[1] + self.size_Title_H + i * self.size_line_H)
            value = indicator_values[id]
            name = self.BarNames[id]

            #myFont = ImageFont.truetype('FreeMono.ttf', 13)
            myFont = ImageFont.load_default()
            draw.text((x,y-8),  f"{name}:", (255, 255, 255), font=myFont)

            line_color = value_to_color(value, self.value_max)

            draw.line((100+x,y, 100+x+5*value,y), width=4, fill=line_color)

        image.save(output_path)






if __name__ == '__main__':
    main()








