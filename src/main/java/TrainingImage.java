import java.util.ArrayList;

public class TrainingImage {
    private final int id;
    private final ArrayList<Float> featureVector;

    public TrainingImage(int id, ArrayList<Float> featureVector) {
        this.id = id;
        this.featureVector = featureVector;
    }

    public int getId() {
        return id;
    }
    public ArrayList<Float> getFeatureVector() {
        return featureVector;
    }
}
