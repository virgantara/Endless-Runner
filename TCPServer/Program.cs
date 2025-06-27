using System;
using System.Net;
using System.Net.Sockets;
using System.Text;
using System.Threading;
using Gtk;

class TcpServerApp
{
    static TcpListener? listener;
    static bool isRunning = false;
    static TextView? logTextView;
    static Thread? serverThread;

    public static void Main()
    {
        Application.Init();

        // === Window Setup ===
        Window win = new Window("TCP Server GTK");
        win.SetDefaultSize(400, 300);
        win.DeleteEvent += delegate { isRunning = false; listener?.Stop(); Application.Quit(); };

        VBox vbox = new VBox(false, 5);

        Button startBtn = new Button("Start Server");
        Button stopBtn = new Button("Stop Server");
        logTextView = new TextView();
        logTextView.Editable = false;

        vbox.PackStart(startBtn, false, false, 0);
        vbox.PackStart(stopBtn, false, false, 0);
        vbox.PackStart(new ScrolledWindow() { logTextView }, true, true, 0);

        win.Add(vbox);
        win.ShowAll();

        // === Button Events ===
        startBtn.Clicked += (s, e) =>
        {
            if (!isRunning)
            {
                isRunning = true;
                serverThread = new Thread(StartServer);
                serverThread.Start();
                AppendLog("Server started.");
            }
        };

        stopBtn.Clicked += (s, e) =>
        {
            isRunning = false;
            listener?.Stop();
            AppendLog("Server stopped.");
        };

        Application.Run();
    }

    static void StartServer()
    {
        int port = 5005;
        listener = new TcpListener(IPAddress.Any, port);
        listener.Start();

        while (isRunning)
        {
            try
            {
                TcpClient client = listener.AcceptTcpClient();
                AppendLog("Client connected.");
                Thread t = new Thread(() => HandleClient(client));
                t.Start();
            }
            catch
            {
                if (!isRunning) break;
            }
        }
    }

    static void HandleClient(TcpClient client)
    {
        NetworkStream stream = client.GetStream();
        byte[] buffer = new byte[1024];

        try
        {
            while (true)
            {
                int byteCount = stream.Read(buffer, 0, buffer.Length);
                if (byteCount == 0) break;
                string message = Encoding.UTF8.GetString(buffer, 0, byteCount);
                AppendLog("Received: " + message);
            }
        }
        catch (Exception ex)
        {
            AppendLog("Error: " + ex.Message);
        }
        finally
        {
            client.Close();
            AppendLog("Client disconnected.");
        }
    }

    static void AppendLog(string message)
	{
	    Application.Invoke(delegate
	    {
	        if (logTextView == null)
	        {
	            Console.WriteLine("logTextView is not initialized.");
	            return;
	        }

	        var buffer = logTextView.Buffer;
	        string timestamped = $"{DateTime.Now:HH:mm:ss} - {message}\n";
	        
	        // Sisipkan log baru di awal teks
	        buffer.Text = timestamped + buffer.Text;

	        // Auto-scroll ke atas (karena log terbaru ada di atas)
	        TextIter startIter = buffer.StartIter;
	        logTextView.ScrollToIter(startIter, 0, false, 0, 0);
	    });
	}


}
