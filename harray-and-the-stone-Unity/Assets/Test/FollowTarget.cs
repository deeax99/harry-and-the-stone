﻿using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class FollowTarget : MonoBehaviour, IAgent
{
    [SerializeField] private Transform target;
    [SerializeField] private float speed = .1f;
    private Vector3 startPosition;
    void Start()
    {
        startPosition = transform.position;
        EnvironmentElements.instance.agents.Add(this);
    }
    public void ApplyAction(EnviornmentAction action)
    {
        transform.position = Vector3.MoveTowards(transform.position, target.position, speed);
    }

    public void ResetState()
    {
        transform.position = startPosition;
    }

    public void UpdateState()
    {
        EnvironmentElements.instance.environmentState.theiveX = Mathf.CeilToInt(transform.position.x / EnvironmentElements.PERCESISION).ToString();
        EnvironmentElements.instance.environmentState.theiveY = Mathf.CeilToInt(transform.position.y / EnvironmentElements.PERCESISION).ToString();
    }
}
