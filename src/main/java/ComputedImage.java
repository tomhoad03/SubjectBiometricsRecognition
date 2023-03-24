import org.openimaj.image.MBFImage;
import org.openimaj.image.pixel.ConnectedComponent;

public class ComputedImage {
    private final int id;
    private final boolean isTraining;
    private final ConnectedComponent component;
    private final MBFImage displayImageA;
    private final MBFImage displayImageB;

    public ComputedImage(int id, boolean isTraining, ConnectedComponent component, MBFImage displayImageA, MBFImage displayImageB) {
        this.id = id;
        this.isTraining = isTraining;
        this.component = component;
        this.displayImageA = displayImageA;
        this.displayImageB = displayImageB;
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

    public MBFImage getDisplayImageA() {
        return displayImageA;
    }

    public MBFImage getDisplayImageB() {
        return displayImageB;
    }
}
