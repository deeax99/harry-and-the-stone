﻿using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public interface IAgent
{
    void ApplyAction(EnviornmentAction action);
    void ResetState();
    void UpdateState();
}
