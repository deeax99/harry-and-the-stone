using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Newtonsoft.Json;
public class MLCommunication
{
    private bool isInitialized = false;
    private TCPCommunication tcpCommunication;
    private int port;
    public MLCommunication(int port)
    {
        tcpCommunication = new TCPCommunication();
        this.port = port;
    }
    void Initialize()
    {
        if (!isInitialized)
        {
            tcpCommunication.InitializationClient(port);
            isInitialized = true;
        }
    }
    public EnviornmentAction GetAction()
    {
        Initialize();
        EnviornmentAction actoin = JsonConvert.DeserializeObject<EnviornmentAction>(tcpCommunication.ReciveData());
        return actoin;
    }
    public void SendAction(EnvironmentState message)
    {
        Initialize();
        string stateJson = JsonConvert.SerializeObject(message);
        tcpCommunication.SendData(stateJson);
    }
}
