import ai.djl.modality.cv.output.Joints;
import org.openimaj.feature.DoubleFV;
import org.openimaj.image.MBFImage;
import org.openimaj.image.pixel.Pixel;

public class ComputedImage {
    private final int id;
    private final MBFImage image;
    private final Pixel centroid;
    private final float[] boundaryDistances;
    private final DoubleFV secondCentralisedMoment;
    private final Joints joints;
    private DoubleFV extractedFeature;

    public ComputedImage(int id, MBFImage image, Pixel centroid, float[] boundaryDistances, DoubleFV secondCentralisedMoment, Joints joints) {
        this.id = id;
        this.image = image;
        this.centroid = centroid;
        this.boundaryDistances = boundaryDistances;
        this.secondCentralisedMoment = secondCentralisedMoment;
        this.joints = joints;
    }

    public int getId() {
        return id;
    }

    public MBFImage getImage() {
        return image;
    }

    public Pixel getCentroid() {
        return centroid;
    }

    public float[] getBoundaryDistances() {
        return boundaryDistances;
    }

    public DoubleFV getSecondCentralisedMoment() {
        return secondCentralisedMoment;
    }

    public Joints getJoints() {
        return joints;
    }

    public DoubleFV getExtractedFeature() {
        return extractedFeature;
    }

    public void setExtractedFeature(DoubleFV extractedFeature) {
        this.extractedFeature = extractedFeature;
    }
}
