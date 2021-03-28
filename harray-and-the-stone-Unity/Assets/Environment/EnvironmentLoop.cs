using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class EnvironmentLoop : MonoBehaviour
{
    bool stop = false;
    private void Update()
    {
        if (stop) return;

        var action = MLCommunication.GetAction();

        if (action.isEnd)
        {
            EnvironmentElements.instance.ResetGame();
        }
        else
        {
            EnvironmentElements.instance.ApplyAction(action);
        }
        var message = EnvironmentElements.instance.GetGameMessage();
        MLCommunication.SendAction(message);
    }
}
