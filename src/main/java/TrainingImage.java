import gnu.trove.list.array.TFloatArrayList;
import org.openimaj.image.MBFImage;

public class TrainingImage {
    private final MBFImage image;
    private final TFloatArrayList featureVector;

    public TrainingImage(MBFImage image, TFloatArrayList featureVector) {
        this.image = image;
        this.featureVector = featureVector;
    }

    public MBFImage getImage() {
        return image;
    }

    public TFloatArrayList getFeatureVector() {
        return featureVector;
    }
}
