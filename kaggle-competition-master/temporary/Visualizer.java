import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.io.RandomAccessFile;

import javax.imageio.ImageIO;

public class Visualizer {

    //  program takes data path and line index as arguments, and assumes both are valid

    public static void main(String[] args) throws IOException {

        //  reads the specified line args[1] in the file with path args[0]

        RandomAccessFile dataFile = new RandomAccessFile(args[0], "r");
        dataFile.seek(Integer.parseInt(args[1]) * (1250 + 1));
        byte[] lineBuffer = new byte[1250];
        dataFile.read(lineBuffer, 0, 1250);
        dataFile.close();

        //  generates a bitmap image for visualization purposes

        BufferedImage image = new BufferedImage(100, 100, BufferedImage.TYPE_BYTE_BINARY);
        for (int i = 0; i < 10000; ++i) { image.setRGB(i % 100, i / 100, ((lineBuffer[i / 8] & (0b1 << (7 - (i % 8)))) == 0b0) ? ~0 : 0); }
        ImageIO.write(image, "png", new File("image.png"));

    }

}
