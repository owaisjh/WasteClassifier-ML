import os
import os.path
import cv2
import glob

Trash_Folder = 'generated_trash'
Extract_Folder = 'extracted_trash'

# Get a list of all captch images we need to process
trash_image_files = glob.glob(os.path.join(Trash_Folder, "*"))
counts = {}

for (i, trash_image_file) in enumerate(trash_image_files):
    print("Status:: Processing Image {}/{}".format(i+1, len(trash_image_files)))

    # Filename is the trash type
    filename = os.path.basename(trash_image_file)
    trash_correct_text = os.path.splitext(filename)[0]

    aaaa=0
    if trash_correct_text[-1]>='0' and trash_correct_text[-1]<='9':
        aaaa=1
        if trash_correct_text[-2]>='0' and trash_correct_text[-2]<='9':
            aaaa=2
            if trash_correct_text[-3]>='0' and trash_correct_text[-3]<='9':
                aaaa=3
    lol=trash_correct_text[0:len(trash_correct_text)-aaaa]

    # Load image and manipulate
    image = cv2.imread(trash_image_file)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.copyMakeBorder(gray, 8, 8, 8, 8, cv2.BORDER_REPLICATE)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

    # find the contours (continuous blobs of pixels)
    contours = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    # contours = contours[0]
    
    letter_regions = []
    # Get rectangle that contains the contour
    (x, y, w, h) = cv2.boundingRect(contours[0])
    letter_regions.append((x, y, w, h))

    x, y, w, h = letter_regions[0]
    # Extract letter from original image with 5 pixel margin around edge
    letter_image = gray[y-2500:y+h+2500, x-2500:x+w+2500]

    # Save image in folder

    save_path = os.path.join(Extract_Folder, lol)

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    # Write letters to a file
    # count = counts.get(trash_correct_text, 1)
    p = os.path.join(save_path, "{}.jpg".format(trash_correct_text))
    cv2.imwrite(p, letter_image)

    # increment the count for the current key
    # counts[trash_correct_text] = count+1
