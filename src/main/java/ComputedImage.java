import org.openimaj.feature.DoubleFV;
import org.openimaj.feature.SparseIntFV;
import org.openimaj.image.MBFImage;
import org.openimaj.image.pixel.ConnectedComponent;

public class ComputedImage {
    private final int id;
    private final boolean isTraining;
    private final ConnectedComponent component;
    private final MBFImage image;
    private DoubleFV extractedFeature;
    private DoubleFV centroid;

    public ComputedImage(int id, boolean isTraining, ConnectedComponent component, MBFImage image) {
        this.id = id;
        this.isTraining = isTraining;
        this.component = component;
        this.image = image;
    }

    public int getId() {
        return id;
    }

    public boolean isTraining() {
        return isTraining;
    }

    public ConnectedComponent getComponent() {
        return component;
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

    public DoubleFV getCentroid() {
        return centroid;
    }

    public void setCentroid(DoubleFV centroid) {
        this.centroid = centroid;
    }
}
