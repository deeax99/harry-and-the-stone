using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System;
public class VisionTest : MonoBehaviour
{
    private Transform pointA, pointB;

    public bool isVisiable = false;
    public int collisionCount = 0;

    const float VISISON_ANGLE = 60;

    public void InitializationVisionTest(Transform pointA , Transform pointB)
    {
        this.pointA = pointA;
        this.pointB = pointB;
    }
    
    public void UpdateLine()
    {
        UpdatePosition();
        UpdateRotation();
        UpdateScale();
        UpdateState();
    }
    private bool isValidCollision(string tag)
    {
        return tag == "wall";
    }
    
    private void OnTriggerEnter2D(Collider2D collision)
    {
        if (isValidCollision(collision.tag))
        {
            collisionCount++;
        }
    }
    private void OnTriggerExit2D(Collider2D collision)
    {
        if (isValidCollision(collision.tag))
        {
            collisionCount--;
        }
    }
    #region Update Transform
    void UpdatePosition()
    {
        transform.position = (pointA.position + pointB.position) / 2;
    }
    void UpdateRotation()
    {
        Vector3 diff = pointA.position - pointB.position;
        float angle = Mathf.Atan2(diff.y, diff.x) * Mathf.Rad2Deg;
        transform.rotation = Quaternion.Euler(0, 0, angle);
    }
    void UpdateScale()
    {
        float distnace = Vector3.Distance(pointA.position, pointB.position);
        transform.localScale = Vector3.right * distnace + new Vector3(0, .01f, 1);
    }
    #endregion
    void UpdateState()
    {
        if (!isVisiable && collisionCount == 0 && IsInsideFOV())
        {
            isVisiable = true;
        }
        else if (isVisiable && (collisionCount > 0 || !IsInsideFOV()))
        {
            isVisiable = false;
        }
    }
    bool IsInsideFOV()
    {
        Vector3 forward = pointA.right;
        Vector3 diff = pointB.position - pointA.position;
        float angle = Vector3.AngleBetween(forward, diff) * Mathf.Rad2Deg;
        return angle < VISISON_ANGLE || angle > 360 - VISISON_ANGLE;
    }
}
