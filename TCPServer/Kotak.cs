namespace TCPServer
{
    public class Kotak
    {
        public double X { get; set; }
        public double Y { get; set; }
        public double Width { get; set; } 
        public double Height { get; set; }

        public Kotak(double x, double y, double width = 50, double height = 50)
        {
            X = x;
            Y = y;
            Width = width;
            Height = height;
        }

        public void Move(string direction, double step = 10)
        {
            switch (direction.ToLower())
            {
                case "left":  X -= step; break;
                case "right": X += step; break;
                case "up":    Y -= step; break;
                case "down":  Y += step; break;
            }
        }
    }
}
