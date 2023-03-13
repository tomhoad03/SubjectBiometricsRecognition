import org.openimaj.data.dataset.VFSGroupDataset;
import org.openimaj.image.FImage;
import org.openimaj.image.ImageUtilities;

import java.io.IOException;
import java.nio.file.Paths;

public class Main {
    public static void main(String[] args) throws IOException {
        final VFSGroupDataset<FImage> training = new VFSGroupDataset<>(Paths.get("").toAbsolutePath() + "\\src\\main\\java\\biometrics\\training", ImageUtilities.FIMAGE_READER);
        final VFSGroupDataset<FImage> testing = new VFSGroupDataset<>(Paths.get("").toAbsolutePath() + "\\src\\main\\java\\biometrics\\testing", ImageUtilities.FIMAGE_READER);
    }
}