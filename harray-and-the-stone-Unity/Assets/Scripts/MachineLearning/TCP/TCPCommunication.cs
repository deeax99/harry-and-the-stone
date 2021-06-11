using System.IO;
using System.Net.Sockets;
using System.Text;
public class TCPCommunication
{
    const string SERVER = "localhost";

    private TcpClient client;
    private NetworkStream networkStream;

    public void InitializationClient(int port)
    {
        client = new TcpClient();
        client.Connect(SERVER, port);
        networkStream = client.GetStream();
    }
    
    public void SendData(string json)
    {
        byte[] writeBuffer = Encoding.UTF8.GetBytes(json + "<EOF>");
        networkStream.Write(writeBuffer, 0, writeBuffer.Length);
    }
    public string ReciveData()
    {
        StringBuilder stringBuilder = new StringBuilder();
        byte[] readBuffer = new byte[1024];
        int counter = -1;
#if UNITY_EDITOR
        counter = 100000;
#endif
        do
        {
            int count = networkStream.Read(readBuffer, 0, 1024);
            string recivedMessage = Encoding.UTF8.GetString(readBuffer, 0, count);
            stringBuilder.Append(recivedMessage);
            counter--;
        }
        while (!IsEndOfFile(stringBuilder) && counter == 0); // <EOF>
        stringBuilder.Remove(stringBuilder.Length - 5, 5);
        return stringBuilder.ToString();
    }
    public bool IsEndOfFile(StringBuilder stringBuilder)
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
