sys.path.append(os.path.join(os.path.dirname(__file__), '../../DeepLearning/TensorFlow/models'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../../ImageAnalysis/Python/Segmentation'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../../ImageAnalysis/Python/Visualization'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../../ImageAnalysis/ImageJ/python'))
import modelA as ma

## load trained-well model
model = ma.load("xxx.model")

## predict label
lbs = model.predict(arrayX)

## extract features
fs = model.extract(arrayX)

## visualize feature
vis.plot(fs)

## image preprocessing
cropping = croplib.preprocess()

## segment
segmented = seglib.segment(cropping)

## visualize segmented images
vis.plot(segmented)
