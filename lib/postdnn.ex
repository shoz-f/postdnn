defmodule PostDNN do
  @moduledoc """
  Post-processing utilities for Deep Neural Network.
  """

  alias PostDNN.NIF

  @doc """
  Execute post processing: nms.

  ## Parameters

    * num_boxes       - number of candidate boxes
    * num_class       - number of category class
    * boxes           - binaries, serialized boxes tensor[`num_boxes`][4]; dtype: float32
    * scores          - binaries, serialized score tensor[`num_boxes`][`num_class`]; dtype: float32
    * opts
      * iou_threshold:   - IOU threshold
      * score_threshold: - score cutoff threshold
      * sigma:           - soft IOU parameter
      * boxrepr:         - type of box representation
         * :center  - center pos and width/height
         * :topleft - top-left pos and width/height
         * :corner  - top-left and bottom-right corner pos
      * label: map       - replace "number" with "name" label according to a map %{0 => "foo", 1 => "baa", ...}
      * label: path      - given a file path, read it and create the label map

  ## Examples

    ```elixir
    non_max_suppression_multi_class(
      Nx.shape(scores), Nx.to_binary(boxes), Nx.to_binary(scores), boxrepr: :corner
    )
    ```
  """
  def non_max_suppression_multi_class({num_boxes, num_class}, boxes, scores, opts \\ []) do
    label = case Keyword.get(opts, :label) do
      map when is_map(map) -> map
      path when is_binary(path) ->
        (for item <- File.stream!(path) do String.trim_trailing(item) end)
        |> Enum.with_index(&{&2, &1})
        |> Enum.into(%{})
      any -> any
    end

    box_repr = case Keyword.get(opts, :boxrepr, :center) do
      :center  -> 0
      :topleft -> 1
      :corner  -> 2
    end

    iou_threshold   = Keyword.get(opts, :iou_threshold, 0.5)
    score_threshold = Keyword.get(opts, :score_threshold, 0.25)
    sigma           = Keyword.get(opts, :sigma, 0.0)

    case NIF.dnn_non_max_suppression_multi_class(num_boxes, box_repr, boxes, num_class, scores, iou_threshold, score_threshold, sigma) do
      {:ok, nil} -> :notfind
      {:ok, result} -> Poison.decode(result) |> labeling(label)
      any -> any
    end
  end

  defp labeling(nms_result, label) when is_map(label) do
    {:ok, result} = nms_result

    {
      :ok,
      Map.keys(result)
        |> Enum.map(&{label[String.to_integer(&1)], result[&1]})
        |> Enum.into(%{})
    }
  end
  defp labeling(nms_result, _),  do: nms_result


  @doc """
  Adjust NMS result to aspect of the input image. (letterbox)
  
  ## Parameters:
  
    * res - NMS result %{}
    * [rx, ry] - aspect ratio of the input image
  """
  def adjust2letterbox(res, [rx, ry] \\ [1.0, 1.0]) do
    Enum.reduce(Map.keys(res), res, fn key,map ->
      Map.update!(map, key, &Enum.map(&1, fn [score, x1, y1, x2, y2] ->
        x1 = if x1 < 0.0, do: 0.0, else: x1
        y1 = if y1 < 0.0, do: 0.0, else: y1
        x2 = if x2 > 1.0, do: 1.0, else: x2
        y2 = if y2 > 1.0, do: 1.0, else: y2
        [score, x1/rx, y1/ry, x2/rx, y2/ry]
      end))
    end)
  end


  @doc """
  Create a list of (x,y) coordinates for mesh grid points - top-left of each grid.
  
  ## Parameters
  
    * shape - tupple {width, height} for overall size.
    * pitches - list of grid spacing.
    * opts
      * :center - return center of each grid.
      * :transpose - return transposed table
      * :normalize - normalize (x,y) cordinate to {0.0..1.0}
      * :rowfirst - change to row scan first. (default: column scan first)
  
  ## Examples
  
    ```
    meshgrid({416,416}, [8,16,32,64], [:center])
    ```
  """
  def meshgrid(shape, pitches, opts \\ [])

  def meshgrid(shape, pitches, opts) when is_list(pitches) do
    Enum.map(pitches, &meshgrid(shape, &1, opts))
    |> Nx.concatenate(axis: 1)
  end

  def meshgrid({w, h}, pitch, opts) when w >= 1 and h >= 1 do
    m = trunc(Float.ceil(h/pitch))
    n = trunc(Float.ceil(w/pitch))

    {scale, pitch} = if :normalize in opts,
        do:   {Nx.tensor([pitch/w, pitch/h]), Nx.tensor([pitch/w, pitch/h])},
        else: {Nx.tensor([pitch, pitch]), Nx.tensor([pitch, pitch])}

    # grid coodinates list
    grid = if :rowfirst in opts do
        (for x <- 0..(n-1), y <- 0..(m-1), do: [x, y])
      else
        (for y <- 0..(m-1), x <- 0..(n-1), do: [x, y])
      end
      |> Nx.tensor(type: {:f, 32})
      |> (&if :center in opts, do: Nx.add(&1, 0.5), else: &1).()
      |> Nx.multiply(scale)

    # pitch list
    pitch = Nx.broadcast(pitch, {m*n, 2})
      |> Nx.as_type({:f, 32})

    Nx.concatenate([grid, pitch], axis: 1)
    |> (&if :transpose in opts, do: Nx.transpose(&1), else: &1).()
  end


  @doc """
  Take records satisfying the predicate function `pred?` from table.
  
  ## Parameters

    * tensor - 2rank tensor (table). each row represents a record.
    * pred? - predicate function to sieve records. a function that returns a rank1
    tensor with '1' in the index position of records to be kept and
    '0' in the index position of those to be discarded.

  ## Examples
  
    ```
    pred? = fn tensor -> Nx.greater(tensor, 0.2) end
    sieve(table, pred?)
    ```
  """
  def sieve(tensor, pred?) do
    # apply the predicate to tensor to get the judgment for each record (row)
    judge = pred?.(tensor)

    # count the number of records for which the judgement was YES(1).
    count = Nx.sum(judge) |> Nx.to_number()

    # take only records for which the judgement is YES.
    index =
      Nx.argsort(judge, direction: :desc)
      |> Nx.slice_along_axis(0, count)

    Nx.take(tensor, index)
  end
end
