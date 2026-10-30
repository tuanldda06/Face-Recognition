"""Usage: python demo.py photo.jpg [cpu|cuda]"""
import sys
from MTCNN.mtcnn import MTCNN

def main():
    if len(sys.argv) < 2:
        raise SystemExit('python demo.py IMAGE_PATH [cpu|cuda]')
    device = sys.argv[2] if len(sys.argv) > 2 else 'cpu'
    result = MTCNN(device=device).detect(sys.argv[1])
    print('Boxes:', result.boxes)
    print('Scores:', result.scores)
    print('Landmarks:', result.landmarks)

if __name__ == '__main__':
    main()
