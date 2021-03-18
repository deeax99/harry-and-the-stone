﻿using System.Collections;
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
    public static EnviornmentAction GetAction(EnvironmentState state)
    {
        Initialize();
        string stateJson = JsonConvert.SerializeObject(state);
        TCPCommunication.SendData(stateJson);
        EnviornmentAction actoin = JsonConvert.DeserializeObject<EnviornmentAction>(TCPCommunication.ReciveData());
        return actoin;
    }
}