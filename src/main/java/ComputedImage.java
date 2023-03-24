import org.openimaj.image.MBFImage;
import org.openimaj.image.pixel.ConnectedComponent;

public class ComputedImage {
    private final int id;
    private final boolean isTraining;
    private final ConnectedComponent personComponent;
    private final MBFImage displayImage;

    public ComputedImage(int id, boolean isTraining, ConnectedComponent personComponent, MBFImage displayImage) {
        this.id = id;
        this.isTraining = isTraining;
        this.personComponent = personComponent;
        this.displayImage = displayImage;
    }

    public int getId() {
        return id;
    }

    public boolean isTraining() {
        return isTraining;
    }

    public ConnectedComponent getPersonComponent() {
        return personComponent;
    }

    public MBFImage getDisplayImage() {
        return displayImage;
    }
}
