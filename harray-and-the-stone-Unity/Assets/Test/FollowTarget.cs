using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class FollowTarget : MonoBehaviour, IAgent
{
    [SerializeField] private Transform target;
    [SerializeField] private float speed = .5f;

    public void ApplyAction(EnviornmentAction action)
    {
        EnvironmentElements.instance.environmentState.theiveX = Mathf.CeilToInt(transform.position.x / .1f).ToString();
        EnvironmentElements.instance.environmentState.theiveY = Mathf.CeilToInt(transform.position.y / .1f).ToString();
    }

    private void Update()
    {
        transform.position = Vector3.MoveTowards(transform.position, target.position, Time.deltaTime * speed);
    }
}
