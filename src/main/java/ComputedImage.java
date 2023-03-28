import org.openimaj.feature.SparseIntFV;
import org.openimaj.image.MBFImage;
import org.openimaj.image.pixel.ConnectedComponent;

public class ComputedImage {
    private final int id;
    private final boolean isTraining;
    private final ConnectedComponent component;
    private final MBFImage image;
    private SparseIntFV extractedFeature;
    private SparseIntFV centroid;

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

    public SparseIntFV getExtractedFeature() {
        return extractedFeature;
    }

    public void setExtractedFeature(SparseIntFV extractedFeature) {
        this.extractedFeature = extractedFeature;
    }

    public SparseIntFV getCentroid() {
        return centroid;
    }

    public void setCentroid(SparseIntFV centroid) {
        this.centroid = centroid;
    }
}
