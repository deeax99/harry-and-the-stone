using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class EnvironmentLoop : MonoBehaviour
{
    bool stop = false;
    private void Update()
    {
        if (stop) return;

        var state = EnvironmentElements.instance.environmentState;
        var action = MLCommunication.GetAction(state);
        EnvironmentElements.instance.ApplyAction(action);
    }
}
