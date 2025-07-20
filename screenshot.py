import mss

def save_screenshot(path="screenshot.png"):
    with mss.mss() as sct:
        sct.shot(output=path)

if __name__ == "__main__":
    save_screenshot()