from PIL import Image
# loading labels and img info 
class ImgInfo:
    def __init__(self, line):
       [num, info] = line.split(':')
       self.name = None
       self.left = 100000
       self.top = 100000
       self.right = 0
       self.bottom = 0
       self.label = ''
       self.set_name(num)
       self.identify_bounding_box_and_label(info)

    def set_name(self, num):
       self.name = num+'.png'
    
    def identify_bounding_box_and_label(self, info):
        #input argument info composes of multiple strings of "height, left, top, width;", delimited by semicolon
        digit_positions = info.split(";")
        for digit_position in digit_positions[:-1]:
            position_info = digit_position.split(",")
            # identify bounding box
            d_height = int(position_info[0])
            d_left = int(position_info[1])
            d_top = int(position_info[2])
            d_width = int(position_info[3])
            d_right = d_left+d_width
            d_bottom = d_top+d_height
            self.left = min(self.left, d_left)
            self.top = min(self.top, d_top)
            self.right = max(self.right, d_right)
            self.bottom = max(self.bottom, d_bottom)
            # identify label
            d_label = position_info[4]       
            if d_label == '10':
                d_label = '0'
            self.label += d_label
        # now we identify the smallest area which covers all the digits in the image
        # expand the bounding box by 30% in both x and y directions
        width = self.right - self.left
        height = self.bottom - self.top
        self.left = int(self.left - width*0.15)
        self.top = int(self.top - height*0.15)
        self.right = int(self.right + width*0.15)
        self.bottom = int(self.bottom + height*0.15)
        

    def __str__(self):
       return "name: "+self.name+" label: "+self.label+" left: "+str(self.left)+" top: "+str(self.top)+" right: "+str(self.right)+" bottom: "+str(self.bottom)      

def crop_images(image_path, image_info_file):
    counter = 0
    with open(image_info_file) as f:
        for line in f:
            counter+=1
            img_info = ImgInfo(line)
            print(img_info) 
            img = Image.open(image_path+img_info.name)
            width, height = img.size
            img2 = img.crop((max(img_info.left, 0), max(img_info.top, 0), min(img_info.right, width), min(img_info.bottom, height)))
            img2 = img2.resize((64, 64), Image.ANTIALIAS)
            img2.save(str(counter)+'_'+img_info.label+'.png')

if __name__=="__main__":
    image_info_file = 'test_labels'
    image_path = '/home/lwy/test_process/images/'
    crop_images(image_path, image_info_file)
