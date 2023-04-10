import org.openimaj.image.pixel.Pixel;

public class PolarPixel {
    public float x;
    public float y;
    public double radius;
    public double angle;

    public PolarPixel(Pixel pixel, Pixel centroid) {
        this.x = pixel.getX();
        this.y = pixel.getY();

        double xDiff = pixel.getX() - centroid.getX(), yDiff = pixel.getY() - centroid.getY();
        double radius = Math.sqrt(Math.pow(xDiff, 2) + Math.pow(yDiff, 2));
        double angle = Math.atan(yDiff / xDiff);

        if (xDiff < 0) {
            angle += Math.PI;
        } else if (xDiff > 0 && yDiff < 0) {
            angle += 2 * Math.PI;
        }
        if (xDiff == 0 && yDiff > 0) {
            angle = Math.PI / 2;
        } else if (xDiff == 0 && yDiff < 0) {
            angle = 3 * (Math.PI / 2);
        } else if (xDiff < 0 && yDiff == 0) {
            angle = Math.PI;
        } else if (xDiff > 0 && yDiff == 0) {
            angle = 0;
        }
        this.radius = radius;
        this.angle = angle;
    }

    public float getX() {
        return x;
    }

    public float getY() {
        return y;
    }

    public double getRadius() {
        return radius;
    }

    public double getAngle() {
        return angle;
    }
}