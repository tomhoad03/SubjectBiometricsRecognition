import org.openimaj.feature.DoubleFV;
import org.openimaj.image.MBFImage;
import org.openimaj.image.pixel.ConnectedComponent;
import org.openimaj.math.geometry.shape.Rectangle;

public class ComputedImage {
    private final int id;
    private final MBFImage image;
    private final float[] boundaryDistances;
    private final DoubleFV secondCentralisedMoment;
    private final double areaRatio;
    private final double aspectRatio;
    private DoubleFV extractedFeature;

    public ComputedImage(int id, ConnectedComponent component, MBFImage image) {
        this.id = id;
        this.image = image;

        this.boundaryDistances = component.calculateBoundaryDistanceFromCentre().toArray();
        this.secondCentralisedMoment = new DoubleFV(component.calculateConvexHull().calculateSecondMomentCentralised()).normaliseFV();

        Rectangle r = component.calculateRegularBoundingBox();
        this.areaRatio = (r.height * r.width) / component.calculateArea();
        this.aspectRatio = component.calculateOrientatedBoundingBoxAspectRatio();
    }

    public int getId() {
        return id;
    }

    public MBFImage getImage() {
        return image;
    }

    public float[] getBoundaryDistances() {
        return boundaryDistances;
    }

    public DoubleFV getSecondCentralisedMoment() {
        return secondCentralisedMoment;
    }

    public double getAreaRatio() {
        return areaRatio;
    }

    public double getAspectRatio() {
        return aspectRatio;
    }

    public double getAspectAreaRatio() {
        return aspectRatio / areaRatio;
    }

    public DoubleFV getExtractedFeature() {
        return extractedFeature;
    }

    public void setExtractedFeature(DoubleFV extractedFeature) {
        this.extractedFeature = extractedFeature;
    }
}
