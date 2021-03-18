using System.IO;
using System.Net.Sockets;
using System.Text;
public class TCPCommunication
{
    const string SERVER = "localhost";
    const int PORT = 7979;

    private static TcpClient client;
    private static NetworkStream networkStream;

    public static void InitializationClient()
    {
        client = new TcpClient();
        client.Connect(SERVER, PORT);
        networkStream = client.GetStream();
    }
    
    public static void SendData(string json)
    {
        byte[] writeBuffer = Encoding.UTF8.GetBytes(json + "<EOF>");
        networkStream.Write(writeBuffer, 0, writeBuffer.Length);
    }
    public static string ReciveData()
    {
        StringBuilder stringBuilder = new StringBuilder();
        byte[] readBuffer = new byte[1024];

        int timeOut = 1000000;

        do
        {
            int count = networkStream.Read(readBuffer, 0, 1024);

            string recivedMessage = Encoding.UTF8.GetString(readBuffer, 0, count);
            stringBuilder.Append(recivedMessage);
            timeOut--;
        }
        while (!IsEndOfFile(stringBuilder) && timeOut > 0); // <EOF>
        stringBuilder.Remove(stringBuilder.Length - 5, 5);
        return stringBuilder.ToString();
    }
    public static bool IsEndOfFile(StringBuilder stringBuilder)
    {
        string EOF = "<EOF>";

        if (stringBuilder.Length > 4)
        {
            int stringLength = stringBuilder.Length;
            for (int i = 0; i < 5; i++)
            {
                if (EOF[i] != stringBuilder[i + stringLength - 5])
                    return false;
            }
            return true;
        }
        return false;

    }
}
