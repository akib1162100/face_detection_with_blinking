import java.io.*;
import java.net.ServerSocket;
import java.net.Socket;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;

import java.awt.AWTException;
import java.awt.Robot;
import java.awt.event.KeyEvent;
import java.awt.event.*;

// Read the full article https://dev.to/mateuszjarzyna/build-your-own-http-server-in-java-in-less-than-one-hour-only-get-method-2k02
public class Server {

    public static void main( String[] args ) throws Exception {
        try (ServerSocket serverSocket = new ServerSocket(8080)) {
            while (true) {
                try (Socket client = serverSocket.accept()) {
                    handleClient(client);
                }
            }
        }
    }

    private static void handleClient(Socket client) throws IOException {
		
		try{
				openDoor();
			}
			catch (AWTException ae) {
				ae.printStackTrace();
			}
			catch (IOException ae) {
				ae.printStackTrace();
			}
			catch (InterruptedException ae) {
				ae.printStackTrace();
			}
			
        BufferedReader br = new BufferedReader(new InputStreamReader(client.getInputStream()));

        StringBuilder requestBuilder = new StringBuilder();
        String line;
        while (!(line = br.readLine()).isBlank()) {
            requestBuilder.append(line + "\r\n");
        }

        String request = requestBuilder.toString();
        String[] requestsLines = request.split("\r\n");
        String[] requestLine = requestsLines[0].split(" ");
        String method = requestLine[0];
        String path = requestLine[1];
        String version = requestLine[2];
        String host = requestsLines[1].split(" ")[1];

        List<String> headers = new ArrayList<>();
        for (int h = 2; h < requestsLines.length; h++) {
            String header = requestsLines[h];
            headers.add(header);
        }

        String accessLog = String.format("Client %s, method %s, path %s, version %s, host %s, headers %s",
                client.toString(), method, path, version, host, headers.toString());
        System.out.println(accessLog);


        Path filePath = getFilePath(path);
        if (Files.exists(filePath)) {
            // file exist
            String contentType = guessContentType(filePath);
            sendResponse(client, "200 OK", contentType, Files.readAllBytes(filePath));
			
        } else {
            // 404
            byte[] notFoundContent = "<h1>Not found :(</h1>".getBytes();
            sendResponse(client, "404 Not Found", "text/html", notFoundContent);
        }

    }
	
	public static void openDoor() throws IOException,
                           AWTException, InterruptedException{
							   
		String line;
		String pidInfo ="";

		Process p =Runtime.getRuntime().exec(System.getenv("windir") +"\\system32\\"+"tasklist.exe");

		BufferedReader input =  new BufferedReader(new InputStreamReader(p.getInputStream()));

		while ((line = input.readLine()) != null) {
			pidInfo+=line; 
		}

		input.close();

		if(pidInfo.contains("zkemnetman.exe"))
		{
			System.out.println("Found");
		}else{
			String command = "C:/Program Files (x86)/ZKTeco/zkemnetman/zkemnetman.exe";
			Runtime run = Runtime.getRuntime();
			run.exec(command);
		}
		
		
        try {
            Thread.sleep(1000);
        }
        catch (InterruptedException e)
        {
            // TODO Auto-generated catch block
            e.printStackTrace();
        }


		
        // Create an instance of Robot class
        Robot robot = new Robot();
 
        // Press keys using robot. A gap of
        // of 500 milli seconds is added after
        // every key press
        // robot.keyPress(KeyEvent.VK_H);
        Thread.sleep(50);
		robot.mouseMove(50,135);
		// press the mouse
		Thread.sleep(50);
        robot.mousePress(InputEvent.BUTTON1_MASK);
		//Thread.sleep(50);
		robot.mouseRelease(InputEvent.BUTTON1_MASK);
		Thread.sleep(50);
		robot.mouseMove(100,280);
		Thread.sleep(50);
		robot.mousePress(InputEvent.BUTTON1_MASK);
		//Thread.sleep(50);
		robot.mouseRelease(InputEvent.BUTTON1_MASK);
		Thread.sleep(50);
		// press the alt key
		robot.keyPress(KeyEvent.VK_ALT);
		//Thread.sleep(50);
		robot.keyRelease(KeyEvent.VK_ALT);
		Thread.sleep(50);
		// go to right (->)
		robot.keyPress(KeyEvent.VK_RIGHT);
		//Thread.sleep(50);
		robot.keyRelease(KeyEvent.VK_RIGHT);
		Thread.sleep(50);
		robot.keyPress(KeyEvent.VK_RIGHT);
		//Thread.sleep(50);
		robot.keyRelease(KeyEvent.VK_RIGHT);
		Thread.sleep(50);
		// press the down arrow
		robot.keyPress(KeyEvent.VK_DOWN);
		robot.keyRelease(KeyEvent.VK_DOWN);
		Thread.sleep(50);
		robot.keyPress(KeyEvent.VK_DOWN);
		robot.keyRelease(KeyEvent.VK_DOWN);
		Thread.sleep(50);
		// press enter
		robot.keyPress(KeyEvent.VK_ENTER);
		Thread.sleep(50);
		
	}

    private static void sendResponse(Socket client, String status, String contentType, byte[] content) throws IOException {
        OutputStream clientOutput = client.getOutputStream();
        clientOutput.write(("HTTP/1.1 \r\n" + status).getBytes());
        clientOutput.write(("ContentType: " + contentType + "\r\n").getBytes());
        clientOutput.write("\r\n".getBytes());
        clientOutput.write(content);
        clientOutput.write("\r\n\r\n".getBytes());
        clientOutput.flush();
        client.close();
    }

    private static Path getFilePath(String path) {
        if ("/".equals(path)) {
            path = "/index.html";
        }

        return Paths.get("/tmp/www", path);
    }

    private static String guessContentType(Path filePath) throws IOException {
        return Files.probeContentType(filePath);
    }

}