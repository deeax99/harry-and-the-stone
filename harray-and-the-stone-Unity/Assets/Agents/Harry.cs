using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Harry : MonoBehaviour , IAgent
{
    [SerializeField] private float speed = .15f;
    [SerializeField] private Transform theive;

    private Vector3 startPosition;

    void Start()
    {
        startPosition = transform.position;
        EnvironmentElements.instance.agents.Add(this);
    }

    public void ApplyAction(EnviornmentAction action)
    {
        switch (action.harryCommand)
        {
            case "left":
                transform.position += Vector3.left * speed;
                break;
            case "right":
                transform.position += Vector3.right * speed;
                break;
            case "up":
                transform.position += Vector3.up * speed;
                break;
            case "down":
                transform.position += Vector3.down * speed;
                break;
        }

    }

    public void ResetState()
    {
        transform.position = startPosition;
    }

    public void UpdateState()
    {
        EnvironmentElements.instance.environmentState.harryX = Mathf.RoundToInt(transform.position.x / EnvironmentElements.PERCESISION).ToString();
        EnvironmentElements.instance.environmentState.harryY = Mathf.RoundToInt(transform.position.y / EnvironmentElements.PERCESISION).ToString();
    }
}
