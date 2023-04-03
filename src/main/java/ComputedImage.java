import org.openimaj.feature.DoubleFV;
import org.openimaj.image.MBFImage;
import org.openimaj.image.pixel.ConnectedComponent;

public class ComputedImage {
    private final int id;
    private final float[] boundaryDistances;
    private final double aspectRatio;
    private final MBFImage image;
    private DoubleFV extractedFeature;

    public ComputedImage(int id, ConnectedComponent component, MBFImage image) {
        this.id = id;
        this.boundaryDistances = component.calculateBoundaryDistanceFromCentre().toArray();
        this.aspectRatio = component.calculateOrientatedBoundingBoxAspectRatio();
        this.image = image;
    }

    public int getId() {
        return id;
    }

    public float[] getBoundaryDistances() {
        return boundaryDistances;
    }

    public double getAspectRatio() {
        return aspectRatio;
    }

    public MBFImage getImage() {
        return image;
    }

    public DoubleFV getExtractedFeature() {
        return extractedFeature;
    }

    public void setExtractedFeature(DoubleFV extractedFeature) {
        this.extractedFeature = extractedFeature;
    }
}
