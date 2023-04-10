import ai.djl.modality.cv.output.Joints;
import org.openimaj.feature.DoubleFV;
import org.openimaj.image.pixel.ConnectedComponent;
import org.openimaj.image.pixel.Pixel;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;

public class ComputedImage {
    private final int id;
    private final ConnectedComponent component;
    private final Joints joints;
    private DoubleFV extractedFeature;

    public ComputedImage(int id, ConnectedComponent component, Joints joints) {
        this.id = id;
        this.component = component;
        this.joints = joints;
    }

    public int getId() {
        return id;
    }

    public void extractFeature() {
        this.extractedFeature = extractSilhouetteFV().concatenate(extractJointsFV());
    }

    // Extract silhouette feature vector
    public DoubleFV extractSilhouetteFV() {
        int maxBins = 60, halfBlankBinSize = 2, count = 0, backCount = 0;
        double[] doubleDistances = new double[maxBins - (halfBlankBinSize * 4)];
        ArrayList<PolarPixel> pixels = new ArrayList<>();
        Pixel centroid = component.calculateCentroidPixel();

        for (Pixel pixel : component.getOuterBoundary()) {
            pixels.add(new PolarPixel(pixel, centroid));
        }
        pixels.sort(Comparator.comparingDouble(o -> o.angle));

        ArrayList<Double> bin = new ArrayList<>();
        for (PolarPixel pixel : pixels) {
            bin.add(pixel.radius);

            if ((pixel.angle > (((2 * Math.PI) / maxBins) * (count + 1))) || pixel == pixels.get(pixels.size() - 1)) {
                double sum = 0;
                for (double value : bin) {
                    sum += value;
                }

                if (count < ((maxBins / 4) - halfBlankBinSize)
                        || (count > ((maxBins / 4) + halfBlankBinSize) && count < ((3 * (maxBins / 4)) - halfBlankBinSize))
                        || count > ((3 * (maxBins / 4)) + halfBlankBinSize)) {
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

    // Extract joints feature vector
    public DoubleFV extractJointsFV() {
        List<Joints.Joint> jointsList = joints.getJoints();
        ArrayList<Double> jointRadii = new ArrayList<>();
        ArrayList<PolarPixel> jointPixels = new ArrayList<>();
        double width = component.calculateRegularBoundingBox().getWidth(), height = component.calculateRegularBoundingBox().getHeight();
        Pixel centroid = component.calculateCentroidPixel();

        for (Joints.Joint joint : jointsList) {
            PolarPixel polarPixel = new PolarPixel(new Pixel((int) (joint.getX() * width), (int) (joint.getY() * height)), centroid);
            jointRadii.add(polarPixel.getRadius());
            jointPixels.add(polarPixel);
        }

        // Invariant features to centroid
        double[] jointsArray = new double[35];
        for (int i = 0; i < 9; i++) {
            jointsArray[i] = jointRadii.get(i);
        }
        Collections.reverse(jointRadii);
        for (int i = 9; i < 15; i++) {
            jointsArray[i] = jointRadii.get(i - 9);
        }

        // Inter-face distances
        jointsArray[15] = Math.sqrt(Math.pow(jointPixels.get(4).getX() - jointPixels.get(2).getX(), 2) + Math.pow(jointPixels.get(4).getY() - jointPixels.get(2).getY(), 2)); // left ear to left eye
        jointsArray[16] = Math.sqrt(Math.pow(jointPixels.get(2).getX() - jointPixels.get(0).getX(), 2) + Math.pow(jointPixels.get(2).getY() - jointPixels.get(0).getY(), 2)); // left eye to nose
        jointsArray[17] = Math.sqrt(Math.pow(jointPixels.get(0).getX() - jointPixels.get(1).getX(), 2) + Math.pow(jointPixels.get(0).getY() - jointPixels.get(1).getY(), 2)); // nose to right eye
        jointsArray[18] = Math.sqrt(Math.pow(jointPixels.get(1).getX() - jointPixels.get(3).getX(), 2) + Math.pow(jointPixels.get(1).getY() - jointPixels.get(3).getY(), 2)); // right eye to right ear
        jointsArray[19] = Math.sqrt(Math.pow(jointPixels.get(4).getX() - jointPixels.get(3).getX(), 2) + Math.pow(jointPixels.get(4).getY() - jointPixels.get(3).getY(), 2)); // ear to ear

        // Body widths
        jointsArray[20] = Math.sqrt(Math.pow(jointPixels.get(6).getX() - jointPixels.get(5).getX(), 2) + Math.pow(jointPixels.get(6).getY() - jointPixels.get(5).getY(), 2)); // shoulder to shoulder
        jointsArray[21] = Math.sqrt(Math.pow(jointPixels.get(8).getX() - jointPixels.get(7).getX(), 2) + Math.pow(jointPixels.get(8).getY() - jointPixels.get(7).getY(), 2)); // elbow to elbow
        jointsArray[22] = Math.sqrt(Math.pow(jointPixels.get(jointPixels.size() - 5).getX() - jointPixels.get(jointPixels.size() - 6).getX(), 2) + Math.pow(jointPixels.get(jointPixels.size() - 5).getY() - jointPixels.get(jointPixels.size() - 6).getY(), 2)); // hip to hip
        jointsArray[23] = Math.sqrt(Math.pow(jointPixels.get(jointPixels.size() - 3).getX() - jointPixels.get(jointPixels.size() - 4).getX(), 2) + Math.pow(jointPixels.get(jointPixels.size() - 3).getY() - jointPixels.get(jointPixels.size() - 4).getY(), 2)); // knee to knee
        jointsArray[24] = Math.sqrt(Math.pow(jointPixels.get(jointPixels.size() - 1).getX() - jointPixels.get(jointPixels.size() - 2).getX(), 2) + Math.pow(jointPixels.get(jointPixels.size() - 1).getY() - jointPixels.get(jointPixels.size() - 2).getY(), 2)); // ankle to ankle

        // Body heights
        jointsArray[25] = Math.sqrt(Math.pow(jointPixels.get(6).getX() - jointPixels.get(8).getX(), 2) + Math.pow(jointPixels.get(6).getY() - jointPixels.get(8).getY(), 2)); // left elbow to left shoulder
        jointsArray[26] = Math.sqrt(Math.pow(jointPixels.get(5).getX() - jointPixels.get(7).getX(), 2) + Math.pow(jointPixels.get(6).getY() - jointPixels.get(7).getY(), 2)); // right elbow to right shoulder
        jointsArray[27] = Math.sqrt(Math.pow(jointPixels.get(6).getX() - jointPixels.get(jointPixels.size() - 6).getX(), 2) + Math.pow(jointPixels.get(6).getY() - jointPixels.get(jointPixels.size() - 6).getY(), 2)); // left hip to left shoulder
        jointsArray[28] = Math.sqrt(Math.pow(jointPixels.get(5).getX() - jointPixels.get(jointPixels.size() - 5).getX(), 2) + Math.pow(jointPixels.get(5).getY() - jointPixels.get(jointPixels.size() - 5).getY(), 2)); // right hip to right shoulder
        jointsArray[29] = Math.sqrt(Math.pow(jointPixels.get(8).getX() - jointPixels.get(jointPixels.size() - 6).getX(), 2) + Math.pow(jointPixels.get(8).getY() - jointPixels.get(jointPixels.size() - 6).getY(), 2)); // left hip to left elbow
        jointsArray[30] = Math.sqrt(Math.pow(jointPixels.get(7).getX() - jointPixels.get(jointPixels.size() - 5).getX(), 2) + Math.pow(jointPixels.get(7).getY() - jointPixels.get(jointPixels.size() - 5).getY(), 2)); // right hip to right elbow
        jointsArray[31] = Math.sqrt(Math.pow(jointPixels.get(jointPixels.size() - 4).getX() - jointPixels.get(jointPixels.size() - 6).getX(), 2) + Math.pow(jointPixels.get(jointPixels.size() - 4).getY() - jointPixels.get(jointPixels.size() - 6).getY(), 2)); // left hip to left knee
        jointsArray[32] = Math.sqrt(Math.pow(jointPixels.get(jointPixels.size() - 3).getX() - jointPixels.get(jointPixels.size() - 5).getX(), 2) + Math.pow(jointPixels.get(jointPixels.size() - 3).getY() - jointPixels.get(jointPixels.size() - 5).getY(), 2)); // right hip to right knee
        jointsArray[33] = Math.sqrt(Math.pow(jointPixels.get(jointPixels.size() - 2).getX() - jointPixels.get(jointPixels.size() - 4).getX(), 2) + Math.pow(jointPixels.get(jointPixels.size() - 2).getY() - jointPixels.get(jointPixels.size() - 4).getY(), 2)); // left knee to left ankle
        jointsArray[34] = Math.sqrt(Math.pow(jointPixels.get(jointPixels.size() - 1).getX() - jointPixels.get(jointPixels.size() - 3).getX(), 2) + Math.pow(jointPixels.get(jointPixels.size() - 1).getY() - jointPixels.get(jointPixels.size() - 3).getY(), 2)); // right knee to right ankle

        return new DoubleFV(jointsArray).normaliseFV();
    }

    public DoubleFV getExtractedFeature() {
        return extractedFeature;
    }
}
