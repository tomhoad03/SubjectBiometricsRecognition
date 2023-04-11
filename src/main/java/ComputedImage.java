import ai.djl.modality.cv.output.Joints;
import org.openimaj.feature.DoubleFV;
import org.openimaj.image.pixel.ConnectedComponent;
import org.openimaj.image.pixel.Pixel;
import org.openimaj.math.geometry.shape.Rectangle;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;

public class ComputedImage {
    private final int id;
    private final ConnectedComponent component;
    private final Joints joints;
    private final double[] temperatureCounts;
    private final DoubleFV extractedFeature;

    /**
     * Creates a feature vector from an image
     * @param id The FV id
     * @param component The person connected component
     * @param joints The list of pixels that represent joints
     * @param temperatureCounts The sums of pixels within each temperature region
     */
    public ComputedImage(int id, ConnectedComponent component, Joints joints, double[] temperatureCounts) {
        this.id = id;
        this.component = component;
        this.joints = joints;
        this.temperatureCounts = temperatureCounts;
        this.extractedFeature = extractSilhouetteFV().concatenate(extractJointsFV()).concatenate(extractTemperaturesFV());
    }

    /**
     * Extracts silhouette feature vector
     * @return The silhouette FV
     */
    public DoubleFV extractSilhouetteFV() {
        int maxBins = 56, halfBlankBinSize = 2, count = 0, backCount = 0;
        double[] doubleDistances = new double[maxBins - (halfBlankBinSize * 4)];
        ArrayList<PolarPixel> pixels = new ArrayList<>();
        Pixel centroid = component.calculateCentroidPixel();

        // Calculates all the boundary pixels as polar coordinates
        for (Pixel pixel : component.getOuterBoundary()) {
            pixels.add(new PolarPixel(calculateDistance(pixel, centroid), calculateAngle(pixel, centroid)));
        }
        pixels.sort(Comparator.comparingDouble(PolarPixel::angle));

        // Creates a histogram from the distances
        ArrayList<Double> bin = new ArrayList<>();
        for (PolarPixel pixel : pixels) {
            bin.add(pixel.radius());

            if ((pixel.angle() > (((2 * Math.PI) / maxBins) * (count + 1))) || pixel == pixels.get(pixels.size() - 1)) {
                double sum = 0;
                for (double value : bin) {
                    sum += value;
                }

                if ((count > halfBlankBinSize && count < ((maxBins / 2) - halfBlankBinSize))
                        || (count > ((maxBins / 2) + halfBlankBinSize) && count < (maxBins - halfBlankBinSize))) {
                    doubleDistances[count - backCount] = sum / bin.size();
                } else {
                    backCount++;
                }
                count++;
                bin.clear();
            }
        }
        return new DoubleFV(doubleDistances).normaliseFV();
    }

    /**
     * Extracts joints feature vector
     * @return The joints FV
     */
    public DoubleFV extractJointsFV() {
        Rectangle boundingBox = component.calculateRegularBoundingBox();
        Pixel centroid = component.calculateCentroidPixel();
        List<Joints.Joint> jointsList = joints.getJoints();
        ArrayList<Pixel> jointPixels = new ArrayList<>();

        // Creates a pose model
        for (Joints.Joint joint : jointsList) {
            Pixel pixel = new Pixel((int) (joint.getX() * boundingBox.getWidth()), (int) (joint.getY() *  boundingBox.getHeight()));
            jointPixels.add(pixel);
        }

        PoseModel poseModel = new PoseModel(jointPixels.get(0), // nose
                jointPixels.get(1), // right eye
                jointPixels.get(2), // left eye
                jointPixels.get(3), // right ear
                jointPixels.get(4), // left ear
                jointPixels.get(5), // right shoulder
                jointPixels.get(6), // left shoulder
                jointPixels.get(7), // right elbow
                jointPixels.get(8), // left elbow
                jointPixels.get(jointPixels.size() - 8), // right wrist*
                jointPixels.get(jointPixels.size() - 7), // left wrist*
                jointPixels.get(jointPixels.size() - 6), // right hip
                jointPixels.get(jointPixels.size() - 5), // left hip
                jointPixels.get(jointPixels.size() - 4), // right knee
                jointPixels.get(jointPixels.size() - 3), // left knee
                jointPixels.get(jointPixels.size() - 2), // right ankle
                jointPixels.get(jointPixels.size() - 1)); // left ankle

        // Invariant features to centroid
        double[] jointsArray = new double[35];
        for (int i = 0; i < 9; i++) {
            jointsArray[i] = calculateDistance(jointPixels.get(i), centroid);
        }
        for (int i = 1; i < 7; i++) {
            jointsArray[i] = calculateDistance(jointPixels.get(jointPixels.size() - i), centroid);
        }

        // Inter-face distances
        jointsArray[15] = calculateDistance(poseModel.leftEar(), poseModel.leftEye());
        jointsArray[16] = calculateDistance(poseModel.leftEye(), poseModel.nose());
        jointsArray[17] = calculateDistance(poseModel.nose(), poseModel.rightEye());
        jointsArray[18] = calculateDistance(poseModel.rightEye(), poseModel.rightEar());
        jointsArray[19] = calculateDistance(poseModel.leftEar(), poseModel.rightEar());

        // Body widths
        jointsArray[20] = calculateDistance(poseModel.leftShoulder(), poseModel.rightShoulder());
        jointsArray[21] = calculateDistance(poseModel.leftElbow(), poseModel.rightElbow());
        jointsArray[22] = calculateDistance(poseModel.leftHip(), poseModel.rightHip());
        jointsArray[23] = calculateDistance(poseModel.leftKnee(), poseModel.rightKnee());
        jointsArray[24] = calculateDistance(poseModel.leftAnkle(), poseModel.rightAnkle());

        // Body heights
        jointsArray[25] = calculateDistance(poseModel.leftElbow(), poseModel.leftShoulder());
        jointsArray[26] = calculateDistance(poseModel.rightElbow(), poseModel.rightShoulder());
        jointsArray[27] = calculateDistance(poseModel.leftHip(), poseModel.leftShoulder());
        jointsArray[28] = calculateDistance(poseModel.rightHip(), poseModel.rightShoulder());
        jointsArray[29] = calculateDistance(poseModel.leftHip(), poseModel.leftElbow());
        jointsArray[30] = calculateDistance(poseModel.rightHip(), poseModel.rightElbow());
        jointsArray[31] = calculateDistance(poseModel.leftHip(), poseModel.leftKnee());
        jointsArray[32] = calculateDistance(poseModel.rightHip(), poseModel.rightKnee());
        jointsArray[33] = calculateDistance(poseModel.leftKnee(), poseModel.leftAnkle());
        jointsArray[34] = calculateDistance(poseModel.rightKnee(), poseModel.rightAnkle());

        return new DoubleFV(jointsArray).normaliseFV();
    }

    /**
     * Extracts temperatures feature vector
     * @return The temperatures FV
     */
    public DoubleFV extractTemperaturesFV() {
        return new DoubleFV(temperatureCounts).normaliseFV();
    }


    /**
     * Calculate the distance between two pixels
     * @param pixelA Pixel A
     * @param pixelB Pixel B
     * @return The distance
     */
    public double calculateDistance(Pixel pixelA, Pixel pixelB) {
        return Math.sqrt(Math.pow(pixelA.getX() - pixelB.getX(), 2) + Math.pow(pixelA.getY() - pixelB.getY(), 2));
    }

    /**
     * Calculate the angle between two pixels
     * @param pixelA Pixel A
     * @param pixelB pixel B
     * @return The angle (radians)
     */
    public double calculateAngle(Pixel pixelA, Pixel pixelB) {
        double xDiff = pixelA.getX() - pixelB.getX(), yDiff = pixelA.getY() - pixelB.getY();
        double angle = Math.atan(yDiff / xDiff);

        if (xDiff < 0) {
            angle += Math.PI;
        } else if (xDiff > 0 && yDiff < 0) {
            angle += 2 * Math.PI;
        }
        if (xDiff == 0 && yDiff > 0) {
            angle = Math.PI / 2;
        } else if (xDiff == 0 && yDiff < 0) {
            angle = 3 * (Math.PI / 2);
        } else if (xDiff < 0 && yDiff == 0) {
            angle = Math.PI;
        } else if (xDiff > 0 && yDiff == 0) {
            angle = 0;
        }
        return angle;
    }

    /**
     * @return The FV id
     */
    public int getId() {
        return id;
    }

    /**
     * @return The FV
     */
    public DoubleFV getExtractedFeature() {
        return extractedFeature;
    }

    /**
     * A record to represent a pixel with polar coordinates
     * @param radius
     * @param angle
     */
    public record PolarPixel(double radius, double angle) { }

    /**
     * A record to represent a pose
     * @param nose
     * @param rightEye
     * @param leftEye
     * @param rightEar
     * @param leftEar
     * @param rightShoulder
     * @param leftShoulder
     * @param rightElbow
     * @param leftElbow
     * @param rightWrist
     * @param leftWrist
     * @param rightHip
     * @param leftHip
     * @param rightKnee
     * @param leftKnee
     * @param rightAnkle
     * @param leftAnkle
     */
    public record PoseModel(Pixel nose, Pixel rightEye, Pixel leftEye, Pixel rightEar, Pixel leftEar, Pixel rightShoulder,
                            Pixel leftShoulder, Pixel rightElbow, Pixel leftElbow, Pixel rightWrist, Pixel leftWrist,
                            Pixel rightHip, Pixel leftHip, Pixel rightKnee, Pixel leftKnee, Pixel rightAnkle,
                            Pixel leftAnkle) { }
}
