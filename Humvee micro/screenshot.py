from PIL import ImageGrab

screenshot_pil = ImageGrab.grab(bbox=(300,200,300+25,200+25))
screenshot_pil.save('Images/Defeat.png')