using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class EnvironmentLoop : MonoBehaviour
{
    bool stop = false;
    private void Update()
    {
        if (stop) return;

        var message = EnvironmentElements.instance.GetGameMessage();
        var action = MLCommunication.GetAction(message);
        if (action.isEnd)
        {
            EnvironmentElements.instance.ResetGame();
        }
        else
        {
            EnvironmentElements.instance.ApplyAction(action);
        }
    }
}
