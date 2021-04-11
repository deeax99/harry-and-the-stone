using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Newtonsoft.Json;
public class MLCommunication
{
    private static bool isInitialized = false;
    static void Initialize()
    {
        if (!isInitialized)
        {
            TCPCommunication.InitializationClient();
            isInitialized = true;
        }

    }
    public static EnviornmentAction GetAction()
    {
        Initialize();
        EnviornmentAction actoin = JsonConvert.DeserializeObject<EnviornmentAction>(TCPCommunication.ReciveData());
        return actoin;
    }
    public static void SendAction(GameMessage message)
    {
        Initialize();
        string stateJson = JsonConvert.SerializeObject(message);
        TCPCommunication.SendData(stateJson);
    }
}
